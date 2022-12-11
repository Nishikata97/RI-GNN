import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


class Data(Dataset):
    def __init__(self, data, item2topic, train_len=None):
        inputs, mask, max_len = handle_data(data[1], train_len) 
        self.inputs = np.asarray(inputs) 
        self.targets = np.asarray(data[2])
        self.mask = np.asarray(mask)
        self.length = len(data[1])
        self.max_len = max_len

        self.item2topic = item2topic

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index] 

        max_n_node = self.max_len
        node = np.unique(u_input) 
        items = node.tolist() + (max_n_node - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        
        adj = np.zeros((max_n_node, max_n_node)) 
        for i in np.arange(len(u_input) - 1): 
            if u_input[i + 1] == 0:  
                break
            
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            '''adj[u][u] = 1 
            adj[v][v] = 1'''
            '''if(self.item2topic[u_input[i]]==self.item2topic[u_input[i+1]]): 
                adj[u][v] = 1'''
            adj[u][v] = 1

        u_sum_in = np.sum(adj, 0)  
        u_sum_in[np.where(u_sum_in == 0)] = 1  
        u_A_in = np.divide(adj, u_sum_in)  

        u_sum_out = np.sum(adj, 1)  
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(adj.transpose(), u_sum_out)  

        u_A = np.concatenate([u_A_in, u_A_out]).transpose()  


        
        adj_sg = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1): 
            for j in np.arange(i, len(u_input) - 1):
                if u_input[j] == 0:  
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[j])[0][0]
                '''adj_sg[u][u] = 1 
                adj_sg[v][v] = 1'''

                '''if (self.item2topic[u_input[i]] == self.item2topic[u_input[j]]): 
                    adj_sg[u][v] = 1
                else:
                    continue'''
                adj_sg[u][v] = 1

        sg_sum_in = np.sum(adj_sg, 0)
        sg_sum_in[np.where(sg_sum_in == 0)] = 1
        sg_in = np.divide(adj_sg, sg_sum_in)

        sg_sum_out = np.sum(adj_sg, 1)
        sg_sum_out[np.where(sg_sum_out == 0)] = 1
        sg_out = np.divide(adj_sg.transpose(), sg_sum_out)
        sg = np.concatenate([sg_in, sg_out]).transpose()


        topics = [self.item2topic[item] for item in items]

        return [torch.tensor(alias_inputs), torch.tensor(u_A), torch.tensor(sg), torch.tensor(items), torch.tensor(topics),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input)]

    def __len__(self):
        return self.length
