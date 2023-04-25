import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .BaseModel import OneTowerBaseModel
from .modules.encoder import NRGMSNewsEncoder
from .modules.encoder import TfmNewsEncoder
from .modules.encoder import AttnUserEncoder
from .multihead_self import MultiHeadSelfAttention
from .modules.encoder import AllBertNewsEncoder

class ACTIVATE_cnn(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.newsEncoder = AllBertNewsEncoder(manager)

        self.title_length = manager.title_length
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()

        self.score = nn.Linear(256, 1)
        nn.init.xavier_normal_(self.score.weight)

        self.CNN_d1 = nn.Conv1d(in_channels=768, out_channels=128,
                                kernel_size=3, dilation=1, padding=1)#dilation卷积核元素之间的间距 空洞卷积
        self.CNN_d2 = nn.Conv1d(in_channels=768, out_channels=128,
                                kernel_size=3, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=768, out_channels=128,
                                kernel_size=3, dilation=3, padding=3)
        self.layerNorm = nn.LayerNorm(manager.hidden_dim)
        self.level = 4
        self.cnn_dim = manager.cnn_dim
        self.seqConv2D = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.interactive=nn.Linear(in_features=20, out_features=128, bias=False)

        self.lstm_net = nn.LSTM(768, 128, num_layers=2, dropout=0.1, bidirectional=True)

        self.attention_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.attention_layer1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )

        self.affine1 = nn.Linear(in_features=5, out_features=256, bias=True)
        self.affine2 = nn.Linear(in_features=256, out_features=1, bias=False)

        self.inputlayer = nn.Linear(in_features=768, out_features=512, bias=True)
        self.hiddenlayer = nn.Linear(in_features=512, out_features=256, bias=True)
        self.outputlayer = nn.Linear(in_features=256, out_features=128, bias=True)

        self.linear = nn.Linear(60,1)
        self.lin1 = nn.Linear(in_features=44, out_features=128, bias=False)
        self.lin2 = nn.Linear(in_features=5, out_features=256, bias=False)
    def _cnn_resnet(self, C): #经过3个卷积核元素之间的间距不同的卷积 并且拼接在一起
        # C [30,20,768]
        C_fin = torch.zeros((*C.shape[:2], self.level,self.cnn_dim), device=C.device)#[30, 20, 4, 768]
        C1= C.transpose(-2, -1)
        C1 = self.CNN_d1(C1).transpose(-2,-1)
        C_fin[:, :, 0, :] = self.ReLU(C1)
        C_fin[:, :, 0, :] = C1#[30, 20, 4, 128]

        C2 = C.transpose(-2, -1)
        C2 = self.CNN_d2(C2).transpose(-2,-1)
        C_fin[:, :, 1, :] = self.ReLU(C2)
        C_fin[:, :, 1, :] = C2

        C3 = C.transpose(-2, -1)
        C3 = self.CNN_d3(C3).transpose(-2,-1)
        C_fin[:, :, 2, :] = self.ReLU(C3)
        C_fin[:, :, 2, :] = C3

        TRC = torch.cat([C1, C2, C3], dim=-2).transpose(-1,-2)#[30, 128, 60]

        return C_fin,C1,TRC#[30, 20, 4, 128]

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
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)#对最后的输出在-1的维度上切分为两块
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)#对隐藏层进行处理
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))#根据hidden层对最后的输出设置不同的权重

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

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))  # 定义线性层w再经过一个tanh激活函数
        nn.init.zeros_(self.affine1.bias)  # 定义线性层的b=0
        nn.init.xavier_uniform_(self.affine2.weight)  # 上面定义的bias=False 说明不需要为其初始化

    def Attention(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))
        a = self.affine2(attention).squeeze(dim=2)
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1)
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)
        out = torch.bmm(alpha, feature).squeeze(dim=1)
        return out

    def infer(self, x):
        #经过Bert embedding的候选新闻
        cdd_news_repr, _ = self._encode_news(x)  #[6, 5, 20, 768] [batch_size,negative-num+1,title_length,hidden_dim]
        self.batch_size = x["cdd_token_id"].shape[0]
        self.cdd_size = x['cdd_token_id'].shape[1]
        his_news_repr, _ = self._encode_news(x, cdd=False)  #[6, 50, 20, 768] [batch_size,his_size,title_length,hidden_dim]
        self.his_size = x["his_token_id"].shape[1]
        cdd_news_repr = cdd_news_repr.view(-1, self.title_length, 768) #[30,20,768]
        his_news_repr = his_news_repr.view(-1, self.title_length, 768) #[300, 20, 768]

        #对候选新闻做Bi-LSTM+CNN
        cdd_cnn, C,TRC= self._cnn_resnet(cdd_news_repr)  # [30, 20, 4, 128],[30,20,128]
        output, (final_hidden_state, final_cell_state) = self.lstm_net(cdd_news_repr.transpose(0, 1))
        output = output.permute(1, 0, 2)  # [batch_size,title_length,num_directions*hidden_size]  [30, 20, 128*2]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)  #[batch_size,num_directions*num_layers,hidden_size] [30, 4, 128]
        cdd_lstm_repr = self.attention_net_with_w(output, final_hidden_state) #[30,128]

        TRC = self.linear(TRC).squeeze(dim=-1)#[30, 128]
        cdd_news = torch.cat((TRC, cdd_lstm_repr), dim=-1).view(self.batch_size,self.cdd_size,-1) #[6,5,256]

        cdd_cnn[:, :, 3, :] = cdd_lstm_repr.view(self.batch_size* self.cdd_size,128).unsqueeze(dim=1).expand_as(C)
        # 维度变换[30,20,4,128]->[6,5, 20, 4, 128]->[6,5,4,20,128]->[6,5,1,4,20,128]
        cdd_cnn = cdd_cnn.view(self.batch_size,self.cdd_size,-1,self.level,self.cnn_dim).transpose(2,3).unsqueeze(dim=2)


        #对历史新闻做Bi-LSTM+CNN
        his_cnn,C1,TRC1= self._cnn_resnet(his_news_repr)
        outputh, (final_hidden_stateh, final_cell_stateh) = self.lstm_net(his_news_repr.transpose(0, 1))
        outputh = outputh.permute(1, 0, 2)
        final_hidden_stateh = final_hidden_stateh.permute(1, 0, 2)
        his_lstm_repr = self.Iattention_net_with_w(outputh, final_hidden_stateh)

        TRC1 = self.linear(TRC1).squeeze(dim=-1)  # [300, 128]
        his_news = torch.cat((TRC1, his_lstm_repr), dim=-1).view(self.batch_size,self.his_size,-1)  # [6,30,256]

        his_cnn[:, :, 3, :] = his_lstm_repr.view(self.batch_size* self.his_size,128).unsqueeze(dim=1).expand_as(C1)
        # 维度变换[300,20,4,128]->[6,50, 20, 4, 128]->[6,50,4,20,128]->[6,1,50,4,20,128]
        his_cnn = his_cnn.view(self.batch_size,self.his_size,-1,self.level,self.cnn_dim).transpose(2,3).unsqueeze(dim=1)


        #----------------------------------------------------------------------------------------对其和候选新闻做细粒度的交互
        matching = cdd_cnn.matmul(his_cnn.transpose(-1, -2))#[6, 5, 50, 4, 20, 20]
        B, C, N, V, L = matching.shape[:-1]
        cnn_input = matching.view(-1, N, V, L*L).transpose(1, 2)  # B*C, V, N, L, L [30, 4, 50, 20,20]
        cnn_output = self.seqConv2D(cnn_input).squeeze(dim=1)#[30, 5, 44]
        cnn_output=self.lin1(cnn_output)

        #----------------------------------------------------------------------------------------对历史新闻和候选新闻做相加的处理
        # 对候选新闻做维度变换
        #  input [6,5,256] -->[6,5,1,256]--->[6,5,50,256]
        cdd_trans = cdd_news.unsqueeze(dim=2).repeat(1, 1, self.his_size, 1)
        #  [6,50,256]-->[6,1,50,256]-->[6,5,50,256]
        his_trans = his_news.unsqueeze(dim=1).repeat(1, self.cdd_size, 1, 1)
        cut = his_trans - cdd_trans  # [6,5,50,256]
        his_news = torch.cat([cdd_trans, cut, his_trans], dim=-1)  # [6, 5, 50, 768]
        # 做拼接后将其送入3个全连接层
        his_news_input = self.inputlayer(his_news)
        his_news_hidden = self.hiddenlayer(his_news_input)
        his_news_output = self.outputlayer(his_news_hidden).view(-1,self.his_size,self.cnn_dim)# [6, 5, 50, 128]
        out = torch.matmul(cnn_output, his_news_output.transpose(-1, -2)).transpose(-1,-2)#[30, 50, 5]
        user_repr = self.Attention(out).unsqueeze(dim=1)  # [30,1,5]

        # 最后再做点乘得到新闻的得分
        cdd_news = cdd_news.view(-1, 256).unsqueeze(dim=1)#[30,1,256]
        sc = cdd_news * self.lin2(user_repr)  # [30, 1, 256]
        logits = self.score(sc).squeeze(dim=-1).squeeze(dim=-1).view(-1,self.cdd_size)

        return logits  # [batch，cdd_size]


    def forward(self,x):#用来计算损失函数
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss