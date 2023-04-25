import os
import math
import torch
import logging
import subprocess
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import scipy.stats as ss
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from utils.util import pack_results, compute_metrics
from .multihead_self import MultiHeadSelfAttention


class BaseModel(nn.Module):
    def __init__(self, manager, name):
        super().__init__()

        self.his_size = manager.his_size
        self.sequence_length = manager.sequence_length
        self.hidden_dim = manager.hidden_dim
        self.device = manager.device
        self.rank = manager.rank
        self.world_size = manager.world_size

        # set all enable_xxx as attributes
        for k,v in vars(manager).items():
            if k.startswith("enable"):
                setattr(self, k, v)
        self.negative_num = manager.negative_num

        if name is None:
            name = type(self).__name__
        if manager.verbose is not None:
            self.name = "-".join([name, manager.verbose])
        else:
            self.name = name

        self.crossEntropy = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(self.name)


    def get_optimizer(self, manager, dataloader_length):
        optimizer = optim.Adam(self.parameters(), lr=manager.learning_rate)

        scheduler = None
        if manager.scheduler == "linear":
            total_steps = dataloader_length * manager.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = round(manager.warmup * total_steps),
                                            num_training_steps = total_steps)

        return optimizer, scheduler


    def _gather_tensors_variable_shape(self, local_tensor):
        """
        gather tensors from all gpus

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            all_tensors: concatenation of local_tensor in each process
        """
        all_tensors = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_tensors, local_tensor)
        all_tensors[self.rank] = local_tensor
        return torch.cat(all_tensors, dim=0)


    def _compute_gate(self, token_id, attn_mask, gate_mask, token_weight):
        """ gating by the weight of each token

        Returns:
            gated_token_ids: [B, K]
            gated_attn_masks: [B, K]
            gated_token_weight: [B, K]
        """
        if gate_mask is not None:
            keep_k_modifier = self.keep_k_modifier * (gate_mask.sum(dim=-1, keepdim=True) < self.k)
            pad_pos = ~((gate_mask + keep_k_modifier).bool())   # B, L
            token_weight = token_weight.masked_fill(pad_pos, -float('inf'))

            gated_token_weight, gated_token_idx = token_weight.topk(self.k)
            gated_token_weight = torch.softmax(gated_token_weight, dim=-1)
            gated_token_id = token_id.gather(dim=-1, index=gated_token_idx)
            gated_attn_mask = attn_mask.gather(dim=-1, index=gated_token_idx)

        # heuristic gate
        else:
            if token_id.dim() == 2:
                gated_token_id = token_id[:, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, 1: self.k + 1]
            else:
                gated_token_id = token_id[:, :, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, :, 1: self.k + 1]
            gated_token_weight = None

        return gated_token_id, gated_attn_mask, gated_token_weight


    @torch.no_grad()
    def dev(self, manager, loaders, log=False):
        self.eval()

        labels, preds = self._dev(manager, loaders)

        if self.rank == 0:
            metrics = compute_metrics(labels, preds, manager.metrics)
            metrics["main"] = metrics["auc"]
            self.logger.info(metrics)
            if log:
                manager._log(self.name, metrics)
        else:
            metrics = None

        if manager.distributed:
            dist.barrier(device_ids=[self.device])

        return metrics


    @torch.no_grad()
    def test(self, manager, loaders, log=False):
        self.eval()

        preds = self._test(manager, loaders)

        if manager.rank == 0:
            save_dir = "data/cache/results/{}/{}/{}".format(self.name, manager.scale, os.path.split(manager.checkpoint)[-1])
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir + "/prediction.txt"

            index = 1
            with open(save_path, "w") as f:
                for pred in preds:
                    array = np.asarray(pred)
                    rank_list = ss.rankdata(1 - array, method="min")
                    line = str(index) + " [" + ",".join([str(i) for i in rank_list]) + "]" + "\n"
                    f.write(line)
                    index += 1
            try:
                subprocess.run(f"zip -j {os.path.join(save_dir, 'prediction.zip')} {save_path}", shell=True)
            except:
                self.logger.warning("Zip Command Not Found! Skip zipping.")
            self.logger.info("written to prediction at {}!".format(save_path))

        if manager.distributed:
            dist.barrier(device_ids=[self.device])



class TwoTowerBaseModel(BaseModel):
    def __init__(self, manager, name=None):
        """
        base class for two tower models (news encoder and user encoder), which we can cache all news and user representations in advance and speed up inference
        """
        super().__init__(manager, name)
        self.title_length = manager.title_length
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.query_inte = nn.Linear(768, 768)
        nn.init.xavier_normal_(self.query_inte.weight)

        self.mu_inte = nn.Linear(768, 768)
        nn.init.xavier_normal_(self.mu_inte.weight)

        self.query_gate = nn.Linear(768, 768)
        nn.init.xavier_normal_(self.query_gate.weight)

        self.trans_gate = nn.Linear(manager.his_size, manager.his_size)
        nn.init.xavier_normal_(self.trans_gate.weight)

        self.CNN_d1 = nn.Conv1d(in_channels=300, out_channels=100,
                                kernel_size=3, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=300, out_channels=100,
                                kernel_size=3, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=300, out_channels=100,
                                kernel_size=3, dilation=3, padding=3)

        self.lstm_net = nn.LSTM(300, 128, num_layers=2, dropout=0.1, bidirectional=True)
        # nn.init.xavier_normal_(self.lstm_net)

        self.attention_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.multihead_self_attention = MultiHeadSelfAttention(768, 16).to(self.device)

        self.attention_layer1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.query_news = nn.Parameter(torch.randn((1, 768), requires_grad=True))
        nn.init.xavier_normal_(self.query_news)

        self.query_news1 = nn.Parameter(torch.randn((1, 768), requires_grad=True))
        nn.init.xavier_normal_(self.query_news1)

    def _cnn_resnet(self, C):
        # C [100,150,20]

        C1 = self.CNN_d1(C)
        RC1 = C1.transpose(-2, -1)
        RC1 = self.ReLU(RC1)

        C2 = self.CNN_d2(C)
        RC2 = C2.transpose(-2, -1)
        RC2 = self.ReLU(RC2)

        C3 = self.CNN_d3(C)
        RC3 = C3.transpose(-2, -1)
        RC3 = self.ReLU(RC3)  # [100,20,150]

        TRC = torch.cat([RC1, RC2, RC3], dim=-1)
        # print(TRC.size())

        return TRC

    def attention_net_with_w(self, lstm_out, lstm_hidden):

        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))

        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def Iattention_net_with_w(self, lstm_out, lstm_hidden):

        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer1(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))

        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """

        # make sure dimension matches
        query = query.expand(key.shape[0], 1, 768)
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)
        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _compute_logits(self, cdd_news_repr, user_repr):
        """ calculate batch of click probabolity

        Args:
            cdd_news_repr: news-level representation, [batch_size, cdd_size, hidden_dim]
            user_repr: user representation, [batch_size, 1, hidden_dim]

        Returns:
            score of each candidate news, [batch_size, cdd_size]
        """
        score = cdd_news_repr.matmul(user_repr.transpose(-2,-1)).squeeze(-1)/math.sqrt(cdd_news_repr.size(-1))
        return score


    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
        news_token_embedding, news_embedding = self.newsEncoder(token_id, attn_mask)
        return news_token_embedding, news_embedding


    def _encode_user(self, x=None, his_news_embedding=None, his_mask=None):
        if x is None:
            user_embedding = self.userEncoder(his_news_embedding, his_mask=his_mask)
        else:
            _, his_news_embedding = self._encode_news(x, cdd=False)
            user_embedding = self.userEncoder(his_news_embedding, his_mask=x["his_mask"].to(self.device))
        return user_embedding,his_news_embedding


    def forward(self, x):
        _, cdd_news_repr = self._encode_news(x)  #torch.Size([2, 5, 768])

        self.batch_size = x["cdd_token_id"].shape[0]
        self.cdd_size = x['cdd_token_id'].shape[1]

        user_repr1,his_news_repr = self._encode_user(x)  #  torch.Size([2, 768,1])  torch.Size([2, 50, 768])

        his_news_repr, h_query = self.multihead_self_attention(his_news_repr)  # torch.Size([20, 50, 128])

        # x的变换 input [20,5,128] -->[20,5,1,128]--->[20,5,50,128]
        cdd_trans = cdd_news_repr.unsqueeze(dim=2).repeat(1, 1, self.his_size, 1)

        # -----v的内部信息流生成-----
        qu_intra = self.query_inte(h_query)  # torch.Size([20, 50, 128])
        mu_intra = self.mu_inte(his_news_repr)  # torch.Size([20, 50,128])
        v = (qu_intra + mu_intra).unsqueeze(dim=1)  # 【20，1,50，128】

        # -----g的外部信息流生成-----
        qu_gate = self.query_gate(h_query).unsqueeze(dim=1)  # torch.Size([20,1, 50, 128])

        qu_gate = qu_gate.repeat(1, self.cdd_size, 1, 1)

        cdd_trans = self.trans_gate(cdd_trans.transpose(-1, -2)).transpose(-1, -2)  # torch.Size([20, 5, 50, 128])
        g = torch.sigmoid(qu_gate + cdd_trans)  # #[20,5,50,128]

        # -----聚合--------
        his_corr = g * v  # torch.Size([20, 5, 50, 128])

        finall_hisnews = his_corr.view(-1, self.his_size, 768)

        user_repr = self._scaled_dp_attention(self.query_news, finall_hisnews, finall_hisnews).view(self.batch_size,
                                                                                                    self.cdd_size,
                                                                                                    768)

        logits = self._compute_logits(cdd_news_repr*user_repr, user_repr1)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss


    def infer(self, x):
        """
        infer logits with cache when evaluating; subclasses may adjust this function in case the user side encoding is different
        """
        cdd_idx = x["cdd_idx"].to(self.device, non_blocking=True)
        his_idx = x["his_idx"].to(self.device, non_blocking=True)


        self.batch_size1 = x["cdd_token_id"].shape[0]
        self.cdd_size1 = x['cdd_token_id'].shape[1]

        cdd_embedding = self.news_embeddings[cdd_idx]
        his_embedding = self.news_embeddings[his_idx]
        user_repr1, his_news_repr = self._encode_user(his_news_embedding=his_embedding, his_mask=x['his_mask'].to(self.device))


        his_news_repr, h_query = self.multihead_self_attention(his_news_repr)  # torch.Size([20, 50, 128])

        # x的变换 input [20,5,128] -->[20,5,1,128]--->[20,5,50,128]
        cdd_trans = cdd_embedding.unsqueeze(dim=2).repeat(1, 1, self.his_size, 1)

        # -----v的内部信息流生成-----
        qu_intra = self.query_inte(h_query)  # torch.Size([20, 50, 128])
        mu_intra = self.mu_inte(his_news_repr)  # torch.Size([20, 50,128])
        v = (qu_intra + mu_intra).unsqueeze(dim=1)  # 【20，1,50，128】

        # -----g的外部信息流生成-----
        qu_gate = self.query_gate(h_query).unsqueeze(dim=1)  # torch.Size([20,1, 50, 128])

        qu_gate = qu_gate.repeat(1, self.cdd_size, 1, 1)

        cdd_trans = self.trans_gate(cdd_trans.transpose(-1, -2)).transpose(-1, -2)  # torch.Size([20, 5, 50, 128])
        g = torch.sigmoid(qu_gate + cdd_trans)  # #[20,5,50,128]

        # -----聚合--------
        his_corr = g * v  # torch.Size([20, 5, 50, 128])

        finall_hisnews = his_corr.view(-1, self.his_size, 768)

        user_repr = self._scaled_dp_attention(self.query_news, finall_hisnews, finall_hisnews).view(self.batch_size1,
                                                                                                    self.cdd_size1,
                                                                                                    768)
        logits = self._compute_logits(cdd_embedding * user_repr, user_repr1)


        return logits


    @torch.no_grad()
    def encode_news(self, manager, loader_news):
        # every process holds the same copy of news embeddings
        news_embeddings = torch.zeros((len(loader_news.dataset), self.hidden_dim), device=self.device)

        # only encode news on the master node to avoid any problems possibly raised by gatherring
        if manager.rank == 0:
            start_idx = end_idx = 0
            for i, x in enumerate(tqdm(loader_news, ncols=80, desc="Encoding News")):
                _, news_embedding = self._encode_news(x)

                end_idx = start_idx + news_embedding.shape[0]
                news_embeddings[start_idx: end_idx] = news_embedding
                start_idx = end_idx

                if manager.debug:
                    if i > 5:
                        break
        # broadcast news embeddings to all gpus
        if manager.distributed:
            dist.broadcast(news_embeddings, 0)

        self.news_embeddings = news_embeddings


    def _dev(self, manager, loaders):
        self.encode_news(manager, loaders["news"])

        impr_indices = []
        masks = []
        labels = []
        preds = []

        for i, x in enumerate(tqdm(loaders["dev"], ncols=80, desc="Predicting")):
            logits = self.infer(x)

            masks.extend(x["cdd_mask"].tolist())
            impr_indices.extend(x["impr_index"].tolist())
            labels.extend(x["label"].tolist())
            preds.extend(logits.tolist())

        if manager.distributed:
            dist.barrier(device_ids=[self.device])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indices, masks, labels, preds))

            if self.rank == 0:
                impr_indices = []
                masks = []
                labels = []
                preds = []
                for output in outputs:
                    impr_indices.extend(output[0])
                    masks.extend(output[1])
                    labels.extend(output[2])
                    preds.extend(output[3])

                masks = np.asarray(masks, dtype=np.bool8)
                labels = np.asarray(labels, dtype=np.int32)
                preds = np.asarray(preds, dtype=np.float32)
                labels, preds = pack_results(impr_indices, masks, labels, preds)

        else:
            masks = np.asarray(masks, dtype=np.bool8)
            labels = np.asarray(labels, dtype=np.int32)
            preds = np.asarray(preds, dtype=np.float32)
            labels, preds = pack_results(impr_indices, masks, labels, preds)

        return labels, preds


    def _test(self, manager, loaders):
        self.encode_news(manager, loaders["news"])

        impr_indices = []
        masks = []
        preds = []

        for i, x in enumerate(tqdm(loaders["test"], ncols=80, desc="Predicting")):
            logits = self.infer(x)

            masks.extend(x["cdd_mask"].tolist())
            impr_indices.extend(x["impr_index"].tolist())
            preds.extend(logits.tolist())

        if manager.distributed:
            dist.barrier(device_ids=[self.device])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indices, masks, preds))

            if self.rank == 0:
                impr_indices = []
                masks = []
                preds = []
                for output in outputs:
                    impr_indices.extend(output[0])
                    masks.extend(output[1])
                    preds.extend(output[2])

                masks = np.asarray(masks, dtype=np.bool8)
                preds = np.asarray(preds, dtype=np.float32)
                preds, = pack_results(impr_indices, masks, preds)

        else:
            masks = np.asarray(masks, dtype=np.bool8)
            preds = np.asarray(preds, dtype=np.float32)
            preds, = pack_results(impr_indices, masks, preds)

        return preds



class OneTowerBaseModel(BaseModel):
    def __init__(self, manager, name=None):
        super().__init__(manager, name)


    @torch.no_grad()
    def _dev(self, manager, loaders):
        impr_indices = []
        masks = []
        labels = []
        preds = []

        for i, x in enumerate(tqdm(loaders["dev"], ncols=80, desc="Predicting")):
            logits = self.infer(x)

            masks.extend(x["cdd_mask"].tolist())
            impr_indices.extend(x["impr_index"].tolist())
            labels.extend(x["label"].tolist())
            preds.extend(logits.tolist())

        if manager.distributed:
            dist.barrier(device_ids=[self.device])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indices, masks, labels, preds))

            if self.rank == 0:
                impr_indices = []
                masks = []
                labels = []
                preds = []
                for output in outputs:
                    impr_indices.extend(output[0])
                    masks.extend(output[1])
                    labels.extend(output[2])
                    preds.extend(output[3])

                masks = np.asarray(masks, dtype=np.bool8)
                labels = np.asarray(labels, dtype=np.int32)
                preds = np.asarray(preds, dtype=np.float32)
                labels, preds = pack_results(impr_indices, masks, labels, preds)

        else:
            masks = np.asarray(masks, dtype=np.bool8)
            labels = np.asarray(labels, dtype=np.int32)
            preds = np.asarray(preds, dtype=np.float32)
            labels, preds = pack_results(impr_indices, masks, labels, preds)

        return labels, preds


    def _test(self, manager, loaders):
        impr_indices = []
        masks = []
        preds = []

        for i, x in enumerate(tqdm(loaders["test"], ncols=80, desc="Predicting")):
            logits = self.infer(x)

            masks.extend(x["cdd_mask"].tolist())
            impr_indices.extend(x["impr_index"].tolist())
            preds.extend(logits.tolist())

        if manager.distributed:
            dist.barrier(device_ids=[self.device])
            outputs = [None for i in range(self.world_size)]
            dist.all_gather_object(outputs, (impr_indices, masks, preds))

            if self.rank == 0:
                impr_indices = []
                masks = []
                preds = []
                for output in outputs:
                    impr_indices.extend(output[0])
                    masks.extend(output[1])
                    preds.extend(output[2])

                masks = np.asarray(masks, dtype=np.bool8)
                preds = np.asarray(preds, dtype=np.float32)
                preds, = pack_results(impr_indices, masks, preds)

        else:
            masks = np.asarray(masks, dtype=np.bool8)
            preds = np.asarray(preds, dtype=np.float32)
            preds, = pack_results(impr_indices, masks, preds)

        return preds