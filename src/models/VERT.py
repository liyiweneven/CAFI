import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .BaseModel import OneTowerBaseModel
from .modules.encoder import NRGMSNewsEncoder
from .modules.encoder import TfmNewsEncoder
from .modules.encoder import AttnUserEncoder
from .multihead_self import MultiHeadSelfAttention_up
from .modules.encoder import AllBertNewsEncoder

class vert(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.newsEncoder = AllBertNewsEncoder(manager)

        self.title_length = manager.title_length
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()
        self.query_inte = nn.Linear(128, 128)
        nn.init.xavier_normal_(self.query_inte.weight)

        self.mu_inte = nn.Linear(128, 128)
        nn.init.xavier_normal_(self.mu_inte.weight)

        self.query_gate = nn.Linear(128,128)
        nn.init.xavier_normal_(self.query_gate.weight)

        self.att = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.att.weight)

        self.score = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.score.weight)

        self.attcdd = nn.Linear(128, 128)
        nn.init.xavier_normal_(self.attcdd.weight)

        self.user = nn.Linear(256, 128)
        nn.init.xavier_normal_(self.user.weight)

        self.trans_gate = nn.Linear(manager.his_size,manager.his_size)
        nn.init.xavier_normal_(self.trans_gate.weight)

        self.lstm_net = nn.LSTM(768, 128, num_layers=2, dropout=0.1, bidirectional=True)
        # nn.init.xavier_normal_(self.lstm_net)
        self.lstm_net2=nn.LSTM(256,128,num_layers=2, dropout=0.1, bidirectional=True)
        self.attention_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.multihead_self_attention = MultiHeadSelfAttention_up(128, 16).to(self.device)


        self.attention_layer1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.query_news = nn.Parameter(torch.randn((1,128), requires_grad=True))
        nn.init.xavier_normal_(self.query_news)

        self.query_news1 = nn.Parameter(torch.randn((1, 128), requires_grad=True))
        nn.init.xavier_normal_(self.query_news1)

        self.lin = nn.Linear(50,1)
        nn.init.xavier_normal_(self.user.weight)

        self.lin1 = nn.Linear(1,50)
        nn.init.xavier_normal_(self.user.weight)

        self.lin2 = nn.Linear(20, 1)
        nn.init.xavier_normal_(self.user.weight)

        self.lin3 = nn.Linear(768, 128)
        nn.init.xavier_normal_(self.user.weight)

        self.lin4 = nn.Linear(256, 128)
        nn.init.xavier_normal_(self.user.weight)


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

        cdd_repr = cdd_news_repr.view(-1,  self.title_length,768)#[30, 20, 768]
        his_repr = his_news_repr.view(-1, self.title_length, 768)  #bert 出来


        #-----------------------------------------------------送入层次Bi-Lstm
        #对输入的新闻标题送人层次注意力
        output1, (final_hidden_state1, final_cell_state1) = self.lstm_net(cdd_repr.transpose(0, 1))#[20, 30, 256]  [4, 30, 128]
        cdd_news_repr1 = self.attention_net_with_w(output1.permute(1, 0, 2), final_hidden_state1.permute(1, 0, 2)).view(self.batch_size, self.cdd_size, 128)
        output2,(final_hidden_state2, final_cell_state2) = self.lstm_net2(output1)#[20, 30, 256]  [4, 30, 128]
        cdd_news_repr2 = self.attention_net_with_w(output2.permute(1, 0, 2), final_hidden_state2.permute(1, 0, 2)).view(self.batch_size, self.cdd_size,128)  # [6,5,128]

        #cdd_news_repr=torch.cat([cdd_news_repr1,cdd_news_repr2],dim=-1)#[6, 5, 256]
        x=nn.Sigmoid()
        cdd_news_repr1=x(cdd_news_repr1)
        cdd_news=cdd_news_repr1*cdd_news_repr2 #[6,5,128]
        cdd_repr=self.lin3(self.lin2(cdd_repr.transpose(-1,-2)).transpose(-1,-2).squeeze(dim=-2)).view(self.batch_size,self.cdd_size,-1)
        cdd_news_repr=torch.cat((cdd_news,cdd_repr),dim=-1)
        cdd_news_repr = self.lin4(cdd_news_repr)

        outputh1, (final_hidden_stateh1, final_cell_stateh1) = self.lstm_net(his_repr.transpose(0, 1))
        his_news_repr1= self.Iattention_net_with_w(outputh1.permute(1, 0, 2),final_hidden_stateh1.permute(1, 0, 2)).view(self.batch_size,self.his_size,128)  # [6, 50, 128]
        outputh2, (final_hidden_stateh2, final_cell_stateh2) = self.lstm_net2(outputh1)
        his_news_repr2= self.Iattention_net_with_w(outputh2.permute(1, 0, 2), final_hidden_stateh2.permute(1, 0, 2)).view(self.batch_size, self.his_size,128)  # [6, 50, 128]

        #his_news_repr = torch.cat([his_news_repr1, his_news_repr2], dim=-1)  # [6, 5, 256]
        his_news_repr1 = x(his_news_repr1)
        his_news = his_news_repr1 * his_news_repr2  # [6,50,128]
        his_repr = self.lin3(self.lin2(his_repr.transpose(-1, -2)).transpose(-1, -2).squeeze(dim=-2)).view(self.batch_size, self.his_size, -1)
        his_news_repr = torch.cat((his_news, his_repr), dim=-1)
        his_news_repr=self.lin4(his_news_repr)

        his_news_repr=self.lin(his_news_repr.transpose(-1,-2)).transpose(-1,-2)
        his_news_repr, h_query = self.multihead_self_attention(his_news_repr)  # torch.Size([20, 50, 128])
        his_news_repr = self.lin1(his_news_repr.transpose(-1, -2)).transpose(-1, -2)

        user_repr = self._scaled_dp_attention(self.query_news, his_news_repr, his_news_repr)

        sc = cdd_news_repr * user_repr
        logits = self.score(sc).squeeze(dim=-1)
        return logits #【batch，cdd_size]


    def forward(self,x):
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss