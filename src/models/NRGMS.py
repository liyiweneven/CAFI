import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .BaseModel import OneTowerBaseModel
from .modules.encoder import NRGMSNewsEncoder
from .modules.encoder import TfmNewsEncoder
from .modules.encoder import AttnUserEncoder
from .multihead_self import MultiHeadSelfAttention
from .modules.encoder import AllBertNewsEncoder

class NRGMS(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.newsEncoder = AllBertNewsEncoder(manager)

        self.title_length = manager.title_length
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.query_inte = nn.Linear(128, 128)
        nn.init.xavier_normal_(self.query_inte.weight)

        self.mu_inte = nn.Linear(128, 128)
        nn.init.xavier_normal_(self.mu_inte.weight)

        self.query_gate = nn.Linear(128,128)
        nn.init.xavier_normal_(self.query_gate.weight)

        self.trans_gate = nn.Linear(manager.his_size,manager.his_size)
        nn.init.xavier_normal_(self.trans_gate.weight)

        self.CNN_d1 = nn.Conv1d(in_channels=300, out_channels=100,
                                kernel_size=3, dilation=1, padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=300, out_channels=100,
                                kernel_size=3, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=300, out_channels=100,
                                kernel_size=3, dilation=3, padding=3)

        # self.cnn = nn.Conv1d(
        #     in_channels=self.embedding_dim,
        #     out_channels=300,  # 150
        #     kernel_size=1
        # )
        # nn.init.xavier_normal_(self.cnn.weight)
        self.lstm_net = nn.LSTM(768, 128, num_layers=2, dropout=0.1, bidirectional=True)
        # nn.init.xavier_normal_(self.lstm_net)

        self.attention_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.multihead_self_attention = MultiHeadSelfAttention(128, 16).to(self.device)


        self.attention_layer1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.query_news = nn.Parameter(torch.randn((1,128), requires_grad=True))
        nn.init.xavier_normal_(self.query_news)

        self.query_news1 = nn.Parameter(torch.randn((1, 128), requires_grad=True))
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


        return TRC

    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
        news_token_embedding, news_embedding = self.newsEncoder(token_id, attn_mask)
        return news_token_embedding, news_embedding

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
        query = query.expand(key.shape[0], 1, 128)
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)
        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])

        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output


    def infer(self, x):
        cdd_news_repr, _ = self._encode_news(x)  # torch.Size([100, 5, 32, 768])

        self.batch_size = x["cdd_token_id"].shape[0]
        self.cdd_size = x['cdd_token_id'].shape[1]

        his_news_repr, _ = self._encode_news(x, cdd=False)  #torch.Size([100, 50, 32, 768])

        self.his_size = x["his_token_id"].shape[1]

        cdd_news_repr = cdd_news_repr.view(-1,  self.title_length,768)
        his_news_repr = his_news_repr.view(-1, self.title_length, 768)


        output, (final_hidden_state, final_cell_state) = self.lstm_net(cdd_news_repr.transpose(0, 1))

        output = output.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)

        cdd_news_repr = self.attention_net_with_w(output, final_hidden_state).view(self.batch_size, self.cdd_size,128)  # [20,5,128]

        outputh, (final_hidden_stateh, final_cell_stateh) = self.lstm_net(his_news_repr.transpose(0, 1))
        outputh = outputh.permute(1, 0, 2)
        final_hidden_stateh = final_hidden_stateh.permute(1, 0, 2)
        his_news_repr = self.Iattention_net_with_w(outputh, final_hidden_stateh).view(self.batch_size, self.his_size,
                                                                                     128)  # [30,50,128]

        # torch.Size([10, 5, 128])
        # torch.Size([10, 50, 128])


        his_news_repr, h_query = self.multihead_self_attention(his_news_repr)  # torch.Size([20, 50, 128])









        # x的变换 input [20,5,128] -->[20,5,1,128]--->[20,5,50,128]
        cdd_trans = cdd_news_repr.unsqueeze(dim=2).repeat(1, 1, self.his_size, 1)  #zzzz

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

        finall_hisnews = his_corr.view(-1, self.his_size, 128)

        user_repr = self._scaled_dp_attention(self.query_news, finall_hisnews, finall_hisnews).view(self.batch_size,
                                                                                                    self.cdd_size,
                                                                                                    128)
        user_repr1 = self._scaled_dp_attention(self.query_news1, his_news_repr, his_news_repr).view(self.batch_size,
                                                                                                    128,
                                                                                                    1)

        # cdd1 = user_repr * cdd_news_repr  # torch.Size([20, 5, 128])
        logits = torch.bmm( user_repr * cdd_news_repr, user_repr1).squeeze(dim=-1)


        return logits


    def forward(self,x):
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss