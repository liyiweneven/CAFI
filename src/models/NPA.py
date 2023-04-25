import torch
import torch.nn as nn
from .BaseModel import OneTowerBaseModel
from .modules.encoder import CnnNewsEncoder
from .modules.encoder import TfmNewsEncoder
from .modules.encoder import AttnUserEncoder
import math

class NPA(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.encoder = CnnNewsEncoder(manager)
        self.uesrencoder = AttnUserEncoder(manager)

    def _click_predictor(self, cdd_news_repr, user_repr):
        """ calculate batch of click probability

        Args:
            cdd_news_repr: tensor of [batch_size, cdd_size, hidden_dim] 候选新闻
            user_repr: tensor of [batch_size, 1, hidden_dim]

        Returns:
            score: tensor of [batch_size, cdd_size]
        """
        scores = torch.bmm(cdd_news_repr, user_repr.transpose(-2, -1)).squeeze(dim=-1)  # [50,5]

        a = torch.sigmoid(scores)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(scores, dim=1)
        else:
            score = torch.sigmoid(scores)

        return a, score
    def _scaled_dp_attention(self, query, key, value):
        """ calculate scaled attended output of values
        计算的候选或者点击新闻自己做的注意力 类似与encoder中的attention

        Args:
            query: tensor of [batch_size, *, query_num, key_dim]
            key: tensor of [batch_size, *, key_num, key_dim]
            value: tensor of [batch_size, *, key_num, value_dim]

        Returns:
            attn_output: tensor of [batch_size, *, query_num, value_dim]
        """


        assert query.shape[-1] == key.shape[-1]#保证query和key的维度匹配
        key = key.transpose(-2, -1) #transpose是numpy下的一个包 交换位置-1和-2位置上的数值

        attn_weights = torch.matmul(query, key) / math.sqrt(query.shape[-1])#torch.matmul代表tensor的乘法 sqrt不能直接引用 需导入math模块
        # print(attn_weights.shape)
        attn_weights = self.softmax(attn_weights)#对求出的权重进行归一化处理

        attn_output = torch.matmul(attn_weights, value)#得到最后attention的表示
        return attn_output
    def infer(self, x):
        cdd_token_id = x["cdd_token_id"].to(self.device)
        cdd_attn_mask = x["cdd_attn_mask"].to(self.device)
        self.batch_size = x["cdd_token_id"].shape[0]
        self.cdd_size = x['cdd_token_id'].shape[1]
        # torch.Size([100, 5, 20, 150]) # torch.Size([100, 5, 150])
        _, cdd_news = self.encoder(cdd_token_id,cdd_attn_mask) # B, C, V, L, D

        # torch.Size([100, 20, 20, 150])  # torch.Size([100, 20, 150])
        his_token_id = x["his_token_id"].to(self.device)
        his_attn_mask = x["his_attn_mask"].to(self.device)
        _, his_news = self.encoder(his_token_id,his_attn_mask) # B, N, V, L, D


        user = self.uesrencoder(his_news) #torch.Size([100, 1, 150])

        a, logits = self._click_predictor(cdd_news, user)  # 调用预测分数函数(新闻表示 用户表示)

        return logits  # [batch，cdd_size]


    def forward(self,x):
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss