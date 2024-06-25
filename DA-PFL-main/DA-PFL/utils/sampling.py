#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch
import copy

def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    print(samples_per_user,shard_per_class)
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False
  
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all
"""
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
"""
def new_noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[]):
    
    # Rate = [0.9,0.1]
#     class_num = [[90, 70, 30],[90, 30, 70],[70, 90, 30],[70, 30, 90],[30, 90, 70],[30, 70, 90]]
    class_num = [[90, 70, 30]]
#     class_num = [[90,70,30],
#                  [30, 70, 90]]
#     [90, 70, 30],[90, 30, 70],[70, 90, 30],[70, 30, 90],[30, 90, 70],[30, 70, 90]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        else:
            idxs_dict[label].append(i)
            count += 1
#         if label < num_classes and label not in idxs_dict.keys():
#             idxs_dict[label] = []
#         if label < num_classes:
#             idxs_dict[label].append(i)
#             count += 1

    #copy idxs_dict for preaparing
    idxs_dict_copy = copy.deepcopy(idxs_dict)
    shard_per_class = int(shard_per_user * num_users / num_classes)
#     samples_per_user = int( count/num_users )
#     print(samples_per_user)
    # whether to sample more test samples per user
#     if (samples_per_user < 100):
#         double = True
#     else:
#         double = False
    
#     if len(rand_set_all) == 0:
#         rand_set_all = list(range(num_classes)) * shard_per_class
#         random.shuffle(rand_set_all)
#         rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
    # e.g. [[0,1,2],[0,1,2]]
#     if len(rand_set_all) == 0:
#         rand_set_all = list(range(shard_per_user)) * num_users
#         rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
    if len(rand_set_all) == 0:
        res = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0]]
        for i in range(0, num_users):
            print(res[i % 24])
            rand_set_all.append(res[i % 24])
#     print(rand_set_all)

    for i in range(num_users):
        rand_set= []
        select_labels = rand_set_all[i% 24]
        print(select_labels)  
        for j in range(len(select_labels)-1):
            # number per class
            # class_samples_per_user = int(Rate[j]*samples_per_user)
#             print(i,"++",i%(len(class_num)),"++",j)
            class_samples_per_user = int(class_num[i%len(class_num)][j])
#             if j == 0: print(class_samples_per_user)
            for k in range(class_samples_per_user):
                if len(idxs_dict[select_labels[j]])>0:
                    idx = np.random.choice(len(idxs_dict[select_labels[j]]),replace=False)
                    rand_set.append(idxs_dict[select_labels[j]][idx])
#                     idxs_dict[select_labels[j]].pop(idx)
                else:
                    idx = np.random.choice(len(idxs_dict_copy[select_labels[j]]),replace=False)
                    rand_set.append(idxs_dict_copy[select_labels[j]][idx])
#                     idxs_dict_copy[select_labels[j]].pop(idx)               
        dict_users[i] = np.array(rand_set,dtype='int64')         
    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)
            
    return dict_users, rand_set_all