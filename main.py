import time
import argparse
import pickle
import os

import torch.backends.cudnn

from model import *
from utils import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Pet_Supplies_5', help='Pet_Supplies_5/ Movies_and_TV_5')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--num_layers', type=int, default=3, help='the number of layers')
parser.add_argument('--n_iter', type=int, default=1)                                    
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--batch_norm', action='store_true', help='batch_norm')
parser.add_argument('--topic_word_num', type=int, default=10, help='num of keywords per topic.')
parser.add_argument('--weight_file', type=str, default='checkpoints/default/0.pth.tar',  help='saved model')

parser.add_argument('--tau', type=float, default=0.2, help='temperature parameter')
parser.add_argument('--beta', type=float, default=0.001, help='The weight for batch_sess_loss.')

opt = parser.parse_args()

def main():
    init_seed(512) 
    opt.weight_file = 'checkpoints/' + opt.dataset + '/' + str(opt.num_layers)

    if opt.dataset == 'Pet_Supplies_5':
        num_node = 30723
        opt.dropout_local = 0.2
    elif opt.dataset == 'Movies_and_TV_5':
        num_node = 46354
        opt.dropout_local = 0.3
    else:
        num_node = 310

    train_data = pickle.load(open('pre_datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('pre_datasets/' + opt.dataset + '/test.txt', 'rb'))

    item_review = pickle.load(open('pre_datasets/' + opt.dataset + '/i_text', 'rb'))
    vocabulary = pickle.load(open('pre_datasets/' + opt.dataset + '/vocabulary', 'rb'))
    topics = pickle.load(open('pre_datasets/' + opt.dataset + '/topic', 'rb'))
    item2topic = pickle.load(open('pre_datasets/' + opt.dataset + '/item2topic', 'rb'))

    print(len(item_review))
    print(len(topics))
    print(len(item2topic))

    i_text = np.array([ii.flatten() for ii in item_review.values()])
    i_text = i_text.astype(np.int32)

    topic = np.array([ii.flatten() for ii in topics.values()])
    topic = topic.astype(np.int32)

    train_data = Data(train_data, item2topic)
    test_data = Data(test_data, item2topic)

    model = trans_to_cuda(CombineGraph(opt, num_node, i_text, vocabulary, topic, item2topic))

    print(opt)
    print(model)
    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_20, mrr_20, hit_10, mrr_10, hit_5, mrr_5 = train_test(model, train_data, test_data)
        if not os.path.exists('checkpoints' + '/' + opt.dataset):
            os.makedirs('checkpoints' + '/' + opt.dataset)
        torch.save(model.state_dict(), opt.weight_file + '_epoch-' + str(epoch) + '.pth_tar')
        flag = 0
        if hit_20 >= best_result[0]:
            best_result[0] = hit_20
            best_epoch[0] = epoch
            flag = 1
        if mrr_20 >= best_result[1]:
            best_result[1] = mrr_20
            best_epoch[1] = epoch
            flag = 1

        if hit_10 >= best_result[2]:
            best_result[2] = hit_10
            best_epoch[2] = epoch
            flag = 1
        if mrr_10 >= best_result[3]:
            best_result[3] = mrr_10
            best_epoch[3] = epoch
            flag = 1

        if hit_5 >= best_result[4]:
            best_result[4] = hit_5
            best_epoch[4] = epoch
            flag = 1
        if mrr_5 >= best_result[5]:
            best_result[5] = mrr_5
            best_epoch[5] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit_20, mrr_20))
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit_10, mrr_10))
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f' % (hit_5, mrr_5))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[2], best_result[3], best_epoch[2], best_epoch[3]))
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[4], best_result[5], best_epoch[4], best_epoch[5]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    writer = SummaryWriter('tensorboard/embeddings')
    writer.add_embedding(model.node_embedding.weight.data)
    writer.close()
    print('Remember to run Tensorboard thru: tensorboard --logdir=tensorboard')
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()
