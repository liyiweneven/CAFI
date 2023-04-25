import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .BaseModel import OneTowerBaseModel

from .multihead_self import MultiHeadSelfAttention_1
from .modules.encoder import AllBertNewsEncoder

class CAUM(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.newsEncoder = AllBertNewsEncoder(manager)

        self.title_length = manager.title_length
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()

        self.score = nn.Linear(400000, 1)
        nn.init.xavier_normal_(self.score.weight)

        self.multihead_self_attention = MultiHeadSelfAttention_1(400, 20).to(self.device)

        self.CNN = nn.Conv1d(in_channels=400, out_channels=400, kernel_size=3, padding=1)
        self.lin=nn.Linear(800,200)
        self.lin1 = nn.Linear(400, 20)
        self.lin3 = nn.Linear(20,200)
        self.lin2 = nn.Linear(768,400)
        self.lin4=nn.Linear(400,1)
        self.lin5 = nn.Linear(50, 1)
    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
        news_token_embedding, news_embedding = self.newsEncoder(token_id, attn_mask)
        return news_token_embedding, news_embedding


    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values

        Args:
            query: tensor of [batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """
        assert query.shape[-1] == key.shape[-1]
        key = key.transpose(-2, -1)
        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])

        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def _click_predictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim] 候选新闻
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """

        scores = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1).squeeze(dim=-1).view(
            self.batch_size, self.cdd_size)  # [50,5]
        a = torch.sigmoid(scores)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(scores, dim=1)
        else:
            score = torch.sigmoid(scores)

        return a, score
    def infer(self, x):
        cdd_news_repr, _ = self._encode_news(x)  # torch.Size([100, 5, 32, 768])

        self.batch_size = x["cdd_token_id"].shape[0]
        self.cdd_size = x['cdd_token_id'].shape[1]

        his_news_repr, _ = self._encode_news(x, cdd=False)

        self.his_size = x["his_token_id"].shape[1]

        cdd_news_repr = cdd_news_repr.view(-1, self.title_length, 768)#[30, 20, 768]
        cdd_news_repr = self.lin2(cdd_news_repr)
        his_news_repr = his_news_repr.view(-1, self.title_length, 768)  #bert 出来 [300, 20, 768]
        his_news_repr = self.lin2(his_news_repr)

        cdd_repr=cdd_news_repr
        #-----------------------------------------------------利用CNN来捕获短期兴趣
        his_cnn = self.CNN(his_news_repr.permute(0,2,1)).permute(0,2,1)#[300,20,768]
        cdd_news_repr = cdd_news_repr.view(self.batch_size,-1, self.title_length, 400)#[6, 5, 20, 768]
        cdd_news_repr=cdd_news_repr.unsqueeze(dim=2).repeat(1,1,self.his_size,1,1)#[6, 5, 50, 20, 768]

        his_cnn = his_cnn.view(self.batch_size,-1, self.title_length, 400)#[6, 50, 20, 768]
        his_cnn=his_cnn.unsqueeze(dim=1).repeat(1,self.cdd_size,1,1,1)#[6, 5, 50, 20, 768]

        his_cnn_repr = torch.cat((his_cnn, cdd_news_repr), -1)

        his_cnn_repr=self.lin(his_cnn_repr)#[6, 5, 50, 20, 768]

        #-----------------------------------------------------利用自注意力来捕获长期兴趣
        cdd_repr=self.lin1(cdd_repr).view(self.batch_size,-1, self.title_length, 20)#[6,5, 20, 48]
        cdd_repr=cdd_repr.unsqueeze(dim=2).repeat(1,1,self.his_size,1,1)

        his_news_repr,_ = self.multihead_self_attention(his_cnn,cdd_repr)

        his_news_repr =self.lin3(his_news_repr.unsqueeze(dim=3).repeat(1, 1, 1,self.title_length,1))  #[6,5,50, 20, 768]

        his=torch.cat((his_cnn_repr,his_news_repr),dim=-1)
        user_repr = self._scaled_dp_attention(cdd_news_repr, his, his)

        cdd_news_repr=self.lin4(cdd_news_repr).squeeze(-1)
        user_repr=self.lin4(user_repr).squeeze(-1)
        cdd_news_repr = self.lin5(cdd_news_repr.view(self.batch_size* self.cdd_size, -1,20).transpose(-1,-2)).transpose(-1,-2)
        user_repr =self.lin5( user_repr.view(self.batch_size* self.cdd_size, -1,20).transpose(-1,-2)).transpose(-1,-2)


        a, logits = self._click_predictor(cdd_news_repr, user_repr)  # 调用预测分数函数(新闻表示 用户表示)

        if x["user_index"] == 592506:
            print(x["user_index"])
            print(x["cdd_idx"])
            print(a)
        return logits  # [batch，cdd_size]

    def forward(self,x):
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss