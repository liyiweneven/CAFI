import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .BaseModel import OneTowerBaseModel
from .modules.encoder import NRGMSNewsEncoder
from .modules.encoder import TfmNewsEncoder
from .modules.encoder import AttnUserEncoder
from .multihead_self import MultiHeadSelfAttention_2,LayerNorm,FeedForwardLayer
from .modules.encoder import AllBertNewsEncoder

class Ab_ATNR(OneTowerBaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.newsEncoder = AllBertNewsEncoder(manager)

        self.title_length = manager.title_length
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()

        self.score = nn.Linear(256, 1)
        nn.init.xavier_normal_(self.score.weight)

        self.CNN_d1 = nn.Conv1d(in_channels=768, out_channels=256,
                                kernel_size=3, dilation=1, padding=1)#dilation卷积核元素之间的间距 空洞卷积
        self.CNN_d2 = nn.Conv1d(in_channels=768, out_channels=256,
                                kernel_size=3, dilation=2, padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=768, out_channels=256,
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
        self.query_news = nn.Parameter(torch.randn((1, 256), requires_grad=True))
        nn.init.xavier_normal_(self.query_news)
        self.multihead_self_attention = MultiHeadSelfAttention_2(256, 16).to(self.device)
        self.layernorm = LayerNorm(256)
        self.feed = FeedForwardLayer(256, 2)
        self.lin= nn.Linear(in_features=384, out_features=256, bias=False)
        self.lin3 = nn.Linear(in_features=5, out_features=20, bias=False)
        self.lin4 = nn.Linear(in_features=50, out_features=20, bias=False)
        self.lin5 = nn.Linear(in_features=20, out_features=5, bias=False)

        self.affine2 = nn.Linear(in_features=256, out_features=1, bias=False)
        self.linear = nn.Linear(60,1)
        self.lin2 = nn.Linear(in_features=128, out_features=256, bias=False)
    def _cnn_resnet(self, C):
        # C [30,20,768]
        C1= C.transpose(-2, -1)
        C1 = self.CNN_d1(C1).transpose(-2,-1)

        C2 = C.transpose(-2, -1)
        C2 = self.CNN_d2(C2).transpose(-2,-1)

        C3 = C.transpose(-2, -1)
        C3 = self.CNN_d3(C3).transpose(-2,-1)

        TRC = torch.cat([C1, C2, C3], dim=-2).transpose(-1,-2)#[30, 128, 60]

        return TRC#[30, 20, 4, 128]

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
        a = self.affine2(feature).squeeze(dim=2)
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

        #对候选新闻做CNN
        TRC= self._cnn_resnet(cdd_news_repr)  #[30, 60, 128]
        output, (final_hidden_state, final_cell_state) = self.lstm_net(cdd_news_repr.transpose(0, 1))
        output = output.permute(1, 0, 2)  # [batch_size,title_length,num_directions*hidden_size]  [30, 20, 128*2]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)  # [batch_size,num_directions*num_layers,hidden_size] [30, 4, 128]
        cdd_lstm_repr = self.attention_net_with_w(output, final_hidden_state)  # [30,128]

        TRC = self.linear(TRC).squeeze(dim=-1)  # [30, 128]
        cdd_cnn = torch.cat((TRC, cdd_lstm_repr), dim=-1).view(self.batch_size, self.cdd_size, -1)  # [6,5,256]
        cdd_cnn=self.lin(cdd_cnn)


        #对历史新闻做CNN
        TRC1= self._cnn_resnet(his_news_repr) #[300, 60, 128]
        outputh, (final_hidden_stateh, final_cell_stateh) = self.lstm_net(his_news_repr.transpose(0, 1))
        outputh = outputh.permute(1, 0, 2)
        final_hidden_stateh = final_hidden_stateh.permute(1, 0, 2)
        his_lstm_repr = self.Iattention_net_with_w(outputh, final_hidden_stateh)

        TRC1 = self.linear(TRC1).squeeze(dim=-1)  # [300, 128]
        his_cnn = torch.cat((TRC1, his_lstm_repr), dim=-1).view(self.batch_size, self.his_size, -1)  # [6,50,256]
        his_cnn = self.lin(his_cnn)


        his_news_repr, _ = self.multihead_self_attention(cdd_cnn, his_cnn)  # [300, 20, 768]
        his_news_repr = self.layernorm(his_news_repr)
        his_news_repr = self.feed(his_news_repr)
        his_news_repr = self.layernorm(his_news_repr)


        user_repr = self._scaled_dp_attention(self.query_news, his_news_repr, his_news_repr).squeeze(
            dim=1)  # [30,  768]
        user_repr = user_repr.unsqueeze(dim=1)  # [30,1,768]

        sc = cdd_cnn * user_repr


        logits = self.score(sc).squeeze(dim=-1)
        return logits  # 【batch，cdd_size]


    def forward(self,x):#用来计算损失函数
        logits = self.infer(x)
        labels = x["label"].to(self.device)
        loss = self.crossEntropy(logits, labels)
        return loss