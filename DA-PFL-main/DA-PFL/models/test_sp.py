# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
import sys

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label

def test_img_local(net_g, dataset, args,idx=None,indd=None, user_idx=-1, idxs=None, class_compose=None, record = None):
    net_g.eval()
    test_loss = 0
    correct = 0

    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf=True
        datatest_new = []
        usr = idx
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append((torch.reshape(torch.tensor(dataset[idx]['x'][j]),(1,28,28)),torch.tensor(dataset[idx]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf=True
        datatest_new = []
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j],dataset[idx]['y'][j]))
    else:
        leaf=False
    
    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=1, shuffle=False) #args.local_bs
    else:
        data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=1,shuffle=False)#args.local_bs
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    correct_arr = np.zeros(3)
    for idx, (data, target) in enumerate(data_loader):
        lab = target.item()
        # print(lab)
        # sys.exit(0)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        for k in record.keys():
            if lab in record[k]:
                correct_arr[k-1] += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
#             print('pred:',y_pred)
#             print('label:',target.data.view_as(y_pred).long().cpu().sum())

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    # print("count:", count)
    # print("class_compose", class_compose)
    # print("sum:", sum(class_compose))
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    count_arr = np.zeros(3)
    acc_arr = np.zeros(3)
    for i in range(3):
        if class_compose[i] != 0:
            acc_arr[i] = correct_arr[i] / class_compose[i] *100.00
    # print("accuracy:", accuracy)
    # print("acc_arr:", acc_arr)
    # sys.exit(0)
    return  accuracy, test_loss, acc_arr

def test_img_local_all(net, args, dataset_test, dict_users_test,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None,  test_class_compose = None,class_compose=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    # prepare class_compose
    record={}
    num_per_type = {}

    for i in range(num_idxxs):
        record[i]= {1:[], 2:[], 3:[]}
        num_per_type[i] = np.zeros(3)
    for i in range(len(class_compose)):
        for j in range(len(class_compose[i])):
            if class_compose[i][j] <= 30:
                record[i][1].append(j)
                num_per_type[i][0] += test_class_compose[i][j]
            elif 30 < class_compose[i][j] <= 80:
                record[i][2].append(j)
                num_per_type[i][1] += test_class_compose[i][j]
            elif 80 < class_compose[i][j]:
                record[i][3].append(j)
                num_per_type[i][2] += test_class_compose[i][j]
    acc_arr = np.zeros((num_idxxs,3))
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            a, b, acc_arr[idx] = test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx, class_compose=num_per_type[idx].tolist(), record = record[idx])
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            a, b, acc_arr[idx] = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx], class_compose=num_per_type[idx].tolist(), record = record[idx])
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = a*len(dict_users_test[idx])
            loss_test_local[idx] = b*len(dict_users_test[idx])
        del net_local
    
    if return_all:
        return acc_test_local, loss_test_local
#     return  sum(acc_test_local)/tot, sum(loss_test_local)/tot
    if 'femnist' in args.dataset or 'sent140' in args.dataset:
        if w_locals != None:
            for i in range(0, args.num_users):
                print("Client {},test acc {:.3f}".format(i,acc_test_local[i]/len(dataset_test[dict_users_test[i]]['x'])))
            print("per class ", np.average(acc_arr, axis=0))
            return sum(acc_test_local)/(tot), sum(loss_test_local)/(tot)
        #         return  sum(acc_test_local)/(num_idxxs), sum(loss_test_local)/(num_idxxs)
        else:
            for i in range(0, args.num_users):
                print("Global model in Client {},test acc {:.3f}".format(i,acc_test_local[i]/len(dataset_test[dict_users_test[i]]['x'])))
            print("per class ", np.average(acc_arr, axis=0))
            return sum(acc_test_local)/(tot), sum(loss_test_local)/(tot)
    else:
        if w_locals != None:
            for i in range(0, args.num_users):
                print("Client {},test acc {:.3f}".format(i,acc_test_local[i]/len(dict_users_test[i])))
            print("per class ", np.average(acc_arr, axis=0))
            return sum(acc_test_local)/(tot), sum(loss_test_local)/(tot)
        #         return  sum(acc_test_local)/(num_idxxs), sum(loss_test_local)/(num_idxxs)
        else:
            for i in range(0, args.num_users):
                print("Global model in Client {},test acc {:.3f}".format(i,acc_test_local[i]/len(dict_users_test[i])))
            print("per class ", np.average(acc_arr, axis=0))
            return sum(acc_test_local)/(tot), sum(loss_test_local)/(tot)
