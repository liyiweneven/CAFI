import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .BaseModel import OneTowerBaseModel

from .multihead_self import MultiHeadSelfAttention_2,LayerNorm,FeedForwardLayer
from .modules.encoder import CnnNewsEncoder

class ATNR(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.encoder = CnnNewsEncoder(manager)
        self.title_length = manager.title_length
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()

        self.score = nn.Linear(5120, 1)
        nn.init.xavier_normal_(self.score.weight)

        self.query_news = nn.Parameter(torch.randn((1, 256), requires_grad=True))
        nn.init.xavier_normal_(self.query_news)
        self.query_news1 = nn.Parameter(torch.randn((1, 256), requires_grad=True))
        nn.init.xavier_normal_(self.query_news1)
        self.multihead_self_attention = MultiHeadSelfAttention_2(256, 16).to(self.device)
        self.layernorm=LayerNorm(256)
        self.feed=FeedForwardLayer(256,2)



    def _encode_news(self, x, cdd=True):
        if cdd:
            token_id = x["cdd_token_id"].to(self.device)
            attn_mask = x['cdd_attn_mask'].to(self.device)
        else:
            token_id = x["his_token_id"].to(self.device)
            attn_mask = x["his_attn_mask"].to(self.device)
        news_token_embedding, news_embedding = self.encoder(token_id, attn_mask)
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


    def infer(self, x):
        cdd_news_repr, _ = self._encode_news(x)  # torch.Size([100, 5, 32, 768])

        self.batch_size = x["cdd_token_id"].shape[0]
        self.cdd_size = x['cdd_token_id'].shape[1]

        his_news_repr, _ = self._encode_news(x, cdd=False)

        self.his_size = x["his_token_id"].shape[1]


        cdd_cnn = cdd_news_repr.view(-1, self.title_length, 256)#[30, 20, 768]
        his_cnn = his_news_repr.view(-1, self.title_length, 256)#[300, 20, 768]

        #-----------------------------------------------------利用自注意力来捕获长期兴趣
        his_news_repr,_ = self.multihead_self_attention(cdd_cnn,his_cnn)#[300, 20, 768]
        his_news_repr = self.layernorm(his_news_repr)
        his_news_repr = self.feed(his_news_repr)
        his_news_repr = self.layernorm(his_news_repr)

        his_news_repr, _ = self.multihead_self_attention(cdd_cnn,his_news_repr )  # [300, 20, 768]
        his_news_repr = self.layernorm(his_news_repr)
        his_news_repr = self.feed(his_news_repr)
        his_news_repr = self.layernorm(his_news_repr)

        his_news_repr, _ = self.multihead_self_attention(cdd_cnn, his_news_repr)  # [300, 20, 768]
        his_news_repr = self.layernorm(his_news_repr)
        his_news_repr = self.feed(his_news_repr)
        his_news_repr = self.layernorm(his_news_repr)

        his_news_repr, _ = self.multihead_self_attention(cdd_cnn, his_news_repr)  # [300, 20, 768]
        his_news_repr = self.layernorm(his_news_repr)
        his_news_repr = self.feed(his_news_repr)
        his_news_repr = self.layernorm(his_news_repr)# [30, 20, 768]


        user_repr = self._scaled_dp_attention(self.query_news, his_news_repr, his_news_repr).squeeze(dim=1)#[30,  768]
        user_repr = user_repr.unsqueeze(dim=1) # [30,1,768]

        cdd_repr = cdd_news_repr.view(-1,self.title_length,256)#[30, 20, 256]

        sc = cdd_repr * user_repr

        sc=sc.reshape(self.batch_size*self.cdd_size,-1)

        logits = self.score(sc).squeeze(dim=-1).view(self.batch_size,self.cdd_size)
        return logits #【batch，cdd_size]


    def forward(self,x):
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss