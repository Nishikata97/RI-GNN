import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from GNN import GGNN, RE_GNN
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class CombineGraph(Module):
    def __init__(self, opt, num_node, i_text, vocabulary, topics, item2topic):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local

        self.num_layers = opt.num_layers

        self.rembedding = nn.Embedding(len(vocabulary), opt.word_dim)

        
        num_topic = len(topics)
        self.topics = nn.Parameter(torch.Tensor(num_topic, opt.word_dim))

        input_dim = opt.hiddenSize
        '''nheads = 3
        self.attentions = [GAT(input_dim, opt.hiddenSize, batch_norm=opt.batch_norm, alpha=opt.alpha, feat_drop=opt.dropout_local) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)'''

        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i % 2 == 0:
                layer = GGNN(input_dim, opt.hiddenSize,
                             batch_norm=opt.batch_norm,
                             feat_drop=opt.dropout_local)
            else:
                layer = RE_GNN(input_dim, opt.hiddenSize, opt.word_dim,
                               batch_norm=opt.batch_norm,
                               dropout_local=opt.dropout_local)
            input_dim += opt.hiddenSize
            self.layers.append(layer)
        self.input_size_final = input_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if opt.batch_norm else None
        self.fc_local = nn.Linear(input_dim, opt.hiddenSize, bias=False) 

        self.i_text = torch.from_numpy(i_text).cuda()

        
        self.node_embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def compute_scores(self, hidden, mask): 
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0] 
        len = hidden.shape[1] 
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1) 

        
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)

        
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1) 
        hs = hs.unsqueeze(-2).repeat(1, len, 1) 

        
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs)) 
        beta = torch.matmul(nh, self.w_2) 
        beta = beta * mask 

        
        select = torch.sum(beta * hidden, 1) 
        

        b = self.node_embedding.weight[1:]  
        scores = torch.matmul(select, b.transpose(1, 0))
        
        return scores 

    def forward(self, items, adj, adj_sg, topics, mask_item, inputs):
        
        batch_size = items.shape[0]
        seqs_len = items.shape[1]
        h = self.node_embedding(items) 
        h_local = h 

        seq_var_word = self.i_text[items] 
        pad_word = self.i_text[0, 0]
        attn_review_mask = seq_var_word != pad_word 
        items_review = self.rembedding(seq_var_word.long()) 

        w_topic = self.topics[topics].unsqueeze(2).repeat(1, 1, items_review.shape[2], 1) 
        items_review = w_topic * items_review 

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                out = layer(h_local, adj)  
            else:
                out = layer(h_local, items_review, adj_sg, attn_review_mask)
            h_local = torch.cat([out, h_local], dim=2)
        h_local = h_local.view(-1, self.input_size_final)
        if self.batch_norm is not None:
            h_local = self.batch_norm(h_local)
        h_local = h_local.view(batch_size, seqs_len, self.input_size_final)

        h_local = self.fc_local(F.dropout(h_local, self.dropout_local, training=self.training)) 

        output = h_local

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, adj_sg, items, topics, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long() 
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    adj_sg = trans_to_cuda(adj_sg).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    topics = trans_to_cuda(topics).long()

    hidden = model(items, adj, adj_sg, topics, mask, inputs) 
    get = lambda index: hidden[index][alias_inputs[index]] 
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss) 
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit_20, mrr_20 = [], []
    hit_10, mrr_10 = [], []
    hit_5, mrr_5 = [], []
    for data in test_loader:
        targets, scores = forward(model, data) 

        sub_20_scores = scores.topk(20)[1] 
        sub_20_scores = trans_to_cpu(sub_20_scores).detach().numpy()
        targets = targets.numpy() 
        for score, target, mask in zip(sub_20_scores, targets, test_data.mask):
            hit_20.append(np.isin(target - 1, score)) 
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1)) 

        sub_10_scores = scores.topk(10)[1]
        sub_10_scores = trans_to_cpu(sub_10_scores).detach().numpy()
        for score, target, mask in zip(sub_10_scores, targets, test_data.mask):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))

        sub_5_scores = scores.topk(5)[1]
        sub_5_scores = trans_to_cpu(sub_5_scores).detach().numpy()
        for score, target, mask in zip(sub_5_scores, targets, test_data.mask):
            hit_5.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_5.append(0)
            else:
                mrr_5.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit_20) * 100)
    result.append(np.mean(mrr_20) * 100)

    result.append(np.mean(hit_10) * 100)
    result.append(np.mean(mrr_10) * 100)

    result.append(np.mean(hit_5) * 100)
    result.append(np.mean(mrr_5) * 100)

    return result
