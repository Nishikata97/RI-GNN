import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

class GGNN(Module):
    def __init__(self, input_size, hidden_size, batch_norm=True, feat_drop=0.0):
        super(GGNN, self).__init__()
        self.input_size = input_size
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.hidden_size = hidden_size 
        self.in_out = hidden_size * 2 
        self.gate_size = 3 * hidden_size 
        
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.in_out)) 
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size)) 
        self.b_ih = Parameter(torch.Tensor(self.gate_size)) 
        self.b_hh = Parameter(torch.Tensor(self.gate_size)) 
        self.b_iah = Parameter(torch.Tensor(self.hidden_size)) 
        self.b_oah = Parameter(torch.Tensor(self.hidden_size)) 

        
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.fc = nn.Linear(input_size, hidden_size, bias=True)

    def forward(self, feat, A):
        
        
        batch_size = feat.shape[0]
        feat = feat.view(-1, self.input_size)
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = feat.view(batch_size, -1, self.input_size)
        hidden = self.fc(self.feat_drop(feat))
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah 
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah 
        inputs = torch.cat([input_in, input_out], 2) 
        gi = F.linear(inputs, self.w_ih, self.b_ih) 
        gh = F.linear(hidden, self.w_hh, self.b_hh) 
        i_r, i_i, i_n = gi.chunk(3, 2) 
        h_r, h_i, h_n = gh.chunk(3, 2)
        
        resetgate = torch.sigmoid(i_r + h_r) 
        inputgate = torch.sigmoid(i_i + h_i) 
        newgate = torch.tanh(i_n + resetgate * h_n) 
        hy = newgate + inputgate * (hidden - newgate) 
        return hy


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = attention_dropout

    def forward(self, q, k, v, scale=None, attn_mask=None):
        
        review_len = q.shape[-2]
        word_dim = q.shape[-1]

        attention = torch.bmm(q.view(-1, review_len, word_dim), k.view(-1, review_len, word_dim).transpose(1, 2))
        if scale:
            attention = attention * scale 

        attention = torch.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        output = torch.bmm(attention, v.view(-1, review_len, word_dim)) 
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, review_dim=300, num_heads=3, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = review_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(review_dim, self.dim_per_head * num_heads) 
        self.linear_k = nn.Linear(review_dim, self.dim_per_head * num_heads) 
        self.linear_v = nn.Linear(review_dim, self.dim_per_head * num_heads) 

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(review_dim, review_dim)
        self.dropout = dropout


    def forward(self, query, key, value, attn_mask=None):
        residual = query 

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.shape[0]
        items_num = key.shape[1]
        review_len = key.shape[2]

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        scale = (key.shape[-1] // num_heads) ** -0.5
        context = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, items_num, review_len, dim_per_head * num_heads)
        output = self.linear_final(context)
        output = F.dropout(output, self.dropout, training=self.training) 
        output = residual + output

        output = torch.sum(output, -2)

        return output 


class RE_GNN(Module): 
    def __init__(self, input_size, hidden_size, word_dim, batch_norm=True, dropout_local=0.0):
        super(RE_GNN, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_local)
        self.dropout = dropout_local
        self.input_size = input_size
        self.hidden_size = hidden_size  
        self.in_out = hidden_size * 2  

        self.fc = nn.Linear(input_size, hidden_size, bias=True)
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size,
                                        bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size,
                                         bias=True)
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))  
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))  
        self.fc_in_out = nn.Linear(self.in_out, hidden_size, bias=True)


        self.attentions = MultiHeadAttention(review_dim=word_dim, num_heads=3, dropout=dropout_local)

        
        review_len = 90
        word_dim = 300
        self.fc_review = nn.Linear(review_len * word_dim, hidden_size,
                                      bias=True)

    def forward(self, h_local, items_review, adj_sg, attn_review_mask):
        
        
        batch_size = h_local.shape[0] 
        N = h_local.shape[1] 

        if self.batch_norm is not None:
            h_local = self.batch_norm(h_local)
        h_local = h_local.view(batch_size, -1, self.input_size)  
        hidden = self.fc(F.dropout(h_local, self.dropout, training=self.training))

        adj_in = adj_sg[:, :, :adj_sg.shape[1]] 
        adj_out = adj_sg[:, :, adj_sg.shape[1]: 2 * adj_sg.shape[1]]

        h_review = self.attentions(items_review, items_review, items_review, attn_review_mask)

        review_sim = torch.cosine_similarity(
            h_review.repeat(1, 1, N).view(batch_size, N, N, -1),
            h_review.repeat(1, N, 1).view(batch_size, N, N, -1),
            dim=-1)  
        zero_vec = -9e15 * torch.ones_like(review_sim)
        sim_in = torch.where(adj_in > 0, review_sim, zero_vec)
        sim_in = F.softmax(sim_in, dim=-1)
        sim_in = F.dropout(sim_in, self.dropout, training=self.training)

        sim_out = torch.where(adj_out > 0, review_sim, zero_vec)
        sim_out = F.softmax(sim_out, dim=-1)
        sim_out = F.dropout(sim_out, self.dropout, training=self.training)

        input_in = torch.matmul(sim_in, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(sim_out, self.linear_edge_out(hidden)) + self.b_oah

        output = self.fc_in_out(torch.cat([input_in, input_out], 2))

        return output
