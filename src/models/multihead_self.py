import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO read


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_a = self.W_Q(Q)
        q_s = q_a.view(batch_size, -1, self.num_attention_heads,self.d_k).transpose(1, 2)

        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(device).expand(batch_size, maxlen) < length.to(device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen,maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_attention_heads,1, 1)
        else:
            attn_mask = None

        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s,attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_attention_heads * self.d_v)
        return context, q_a


class ScaledDotProductAttention_up(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention_up, self).__init__()
        self.d_k = d_k
        self.lin=nn.Linear(50, 1)
    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        c=torch.ge(scores,0)
        d=float('-inf')
        dim0, dim1,dim2,dim3 = c.shape
        for i in range(dim0):
            for j in range(dim1):
                for k in range(dim2):
                    for s in range(dim3):
                        element = c[i][j][k][s]
                        sco = scores[i][j][k][s]
                        if element is False:
                            sco=sco*1
                            scores[i][j][k][s]=sco
                        else:
                            sco=d
                            scores[i][j][k][s] = sco
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention_up(nn.Module):
    def __init__(self, d_model, num_attention_heads): #128 16
        super(MultiHeadSelfAttention_up, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_a = self.W_Q(Q)
        q_s = q_a.view(batch_size, -1, self.num_attention_heads,self.d_k).transpose(1, 2)

        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(device).expand(
                batch_size, maxlen) < length.to(device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen,
                                                      maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_attention_heads,1, 1)
        else:
            attn_mask = None

        context, attn = ScaledDotProductAttention_up(self.d_k)(q_s, k_s, v_s,attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        return context, q_a


class ScaledDotProductAttention_1(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention_1, self).__init__()
        self.d_k = d_k
        self.lin=nn.Linear(400,1).to(device)
        self.lin1=nn.Linear(20,1).to(device)

    def forward(self, Q, K, V, cdd):
        self.batch_size = Q.shape[0]
        self.cdd_length=Q.shape[1]
        self.his_length = Q.shape[2]
        self.length=Q.shape[4]


        Q=self.lin(Q.view(self.batch_size,self.cdd_length,self.his_length,-1,20).transpose(-1,-2)).squeeze(dim=-1)#[6, 5, 50, 20, 16, 48]
        K=self.lin(K.view(self.batch_size,self.cdd_length,self.his_length,-1,20).transpose(-1,-2)).squeeze(dim=-1)
        V=self.lin(V.view(self.batch_size,self.cdd_length,self.his_length,-1,20).transpose(-1,-2)).squeeze(dim=-1)

        cdd = self.lin1(cdd.view(self.batch_size,self.cdd_length,self.his_length,-1,20).transpose(-1,-2)).squeeze(dim=-1)#[6, 5, 50, 20, 48]

        scores1 = torch.matmul(Q, K.transpose(-1, -2))/ np.sqrt(self.d_k)
        scores2 = torch.matmul(K, cdd.transpose(-1, -2))/ np.sqrt(self.d_k)
        scores=scores1+scores2
        scores = torch.exp(scores)
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention_1(nn.Module):
    def __init__(self, d_model, num_attention_heads): #128 16
        super(MultiHeadSelfAttention_1, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, cdd):
        K = Q
        V = Q
        batch_size = Q.size(0)
        cdd_length = Q.size(1)
        his_length = Q.size(2)
        q_a = self.W_Q(Q)
        q_s = q_a.view(batch_size,cdd_length,his_length,-1, self.num_attention_heads,self.d_k)

        k_s = self.W_K(K).view(batch_size,cdd_length,his_length,-1, self.num_attention_heads,self.d_k)

        v_s = self.W_V(V).view(batch_size,cdd_length,his_length,-1, self.num_attention_heads,self.d_k)
        context, attn = ScaledDotProductAttention_1(self.d_k)(q_s, k_s, v_s,cdd)

        return context, q_a

class ScaledDotProductAttention_2(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention_2, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        self.batch_size = Q.shape[0]
        self.length=Q.shape[1]


        scores = torch.matmul(Q, K.transpose(-1, -2))/ np.sqrt(self.d_k)
        scores = torch.exp(scores)
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention_2(nn.Module):
    def __init__(self, d_model, num_attention_heads): #128 16
        super(MultiHeadSelfAttention_2, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, cdd, his):
        K = his
        V = his
        batch_size = cdd.size(0)

        q_a = self.W_Q(cdd)
        q_s = q_a.view(batch_size, -1, self.num_attention_heads,self.d_k).transpose(1, 2)

        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)
        context, attn = ScaledDotProductAttention_2(self.d_k)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        return context, q_a

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):  # embedding_dim: 是一个size, 例如[batch_size, len, embedding_dim], 也可以是embedding_dim。。
        super(LayerNorm, self).__init__()
        # 用 parameter 封装，代表模型的参数，作为调节因子
        self.a = nn.Parameter(torch.ones(embedding_dim))
        self.b = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        # 其实就是对最后一维做标准化
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x-mean) / (std+self.eps) + self.b

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, forward_expansion):
        super(FeedForwardLayer, self).__init__()
        self.w1 = nn.Linear(d_model, d_model*forward_expansion)
        self.w2 = nn.Linear(d_model*forward_expansion, d_model)

    def forward(self, x):
        return self.w2((F.relu(self.w1(x))))


class ScaledDotProductAttention_decay(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention_decay, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)

        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadSelfAttention_decay(nn.Module):
    def __init__(self, d_model, num_attention_heads): #128 16
        super(MultiHeadSelfAttention_decay, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_a = self.W_Q(Q)
        q_s = q_a.view(batch_size, -1, self.num_attention_heads,self.d_k).transpose(1, 2)

        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,self.d_k).transpose(1, 2)

        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(device).expand(
                batch_size, maxlen) < length.to(device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen,maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_attention_heads,1, 1)
        else:
            attn_mask = None

        context, attn = ScaledDotProductAttention_decay(self.d_k)(q_s, k_s, v_s,attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        return context, q_a
