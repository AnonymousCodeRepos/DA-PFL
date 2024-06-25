#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch
import copy
from utils.weighting import get_weight
import json
"""
    Sample non-I.I.D client data from Cifar10 dataset
    :param dataset:
    :param num_users:
    :return:
"""


def new_noniid(dataset, num_users, class_per_user, num_classes, args, user_info_static={}):
    #     class_num = [[90, 70, 30]]
    train = False
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
#     print('labels: ',idxs_dict.keys())
    # copy idxs_dict for preaparing
    idxs_dict_copy = copy.deepcopy(idxs_dict)
#     shard_per_class = int(class_per_user * num_users / num_classes)
#     # the number of per user samples
#     if args.dataset == 'cifar10':
#         user_samples_num = np.random.lognormal(4, 1, (num_users)) + 100
#     if args.dataset == 'cifar100':
#         user_samples_num = np.random.lognormal(9, 1, (num_users)) + 600
    cal_user_weight = np.zeros((num_users, num_users))
    user_class_comp = np.zeros((num_users, num_classes))

    diri_alpha_list = []
    diri_alpha_list.append(args.diri_alpha)

    if len(user_info_static) == 0:
        train = True
        dirichlet = np.random.dirichlet(diri_alpha_list * args.num_users, args.num_classes)
        if args.dataset == 'cifar10':
            dirichlet = dirichlet * (args.num_users * 30)
        elif args.dataset == 'cifar100':
            if args.num_users <100:
                dirichlet = dirichlet * (args.num_users * 30)
            else:
                dirichlet = dirichlet * (args.num_users * 50)
        elif args.dataset == 'mnist':
            dirichlet = dirichlet * (args.num_users * 30)
        elif args.dataset == 'imagenet':
            if args.num_users <100:
                dirichlet = dirichlet * (args.num_users * 60)
            else:
                dirichlet = dirichlet * (args.num_users * 40)
        # transpose dirichlet array --> N*K(client_num * class_num)
        _tmp = dirichlet.T
        cla_num_per_client = []
        # 对dirichlet矩阵进行进一步的处理，满足 imbalance ratio设置
        # int()
        for t in _tmp:
            t = [int(tt) for tt in t]
            cla_num_per_client.append(t)
        # find max/min value
        for lenc in range(len(cla_num_per_client)):
            c = np.array(cla_num_per_client[lenc])
            print(c)
            c_max = c.max()
            if c_max < 100:
                for lc in range(len(c)):
                    if c[lc] == c_max:
                        c[lc] = 120   
                        c_max = 120
            # c_min = np.min(c[np.nonzero(c)])
            aj_min = int(c_max / args.imbalance_ratio)
            class_num_sort = np.sort(c[np.nonzero(c)])
            len_class_num_sort = len(class_num_sort)
            # 如果类的数量<=少数类数量，进行补充，补充到少数类+1
            if len_class_num_sort <= args.min_class:
                for ajc in range(len(c)*100):
                    ran = np.random.choice(args.num_classes)
                    if c[ran] == 0:
                        c[ran] = aj_min
                    else: 
                        continue
                    class_num_sort = np.sort(c[np.nonzero(c)])
                    len_class_num_sort = len(class_num_sort)
                    if len_class_num_sort > args.min_class:
                        break
            print(c)            
            class_num_sort_arr = class_num_sort[0: args.min_class]
            # 这里会出现几种情况（多个类别imbalance后面处理）1少数类：
            # 1）最小值<最大值*比例，那么大部分进行调整，避免其他的类别也因为数据量少产生影响；
            # 2）最小值=>最大值*比例，那么只调整最小值
            # 多类别少数类的处理同上，这里采用最简单的方式，多个少数类同时调整到设定，其他类按照比例调整。
            # 1）少数类列表最大值<最大值*比例，那么大部分进行调整（imb*最大值/少数类最大值），
            #  防止调整后不为少数类，需要注意少数类里面的最大值不超过imb设定
            # 2）少数类列表中间值~最大值*比例，全部变为最大值*比例  3）少数类列表最小值=>最大值*比例，全部变为最大值*比例  合并为一个
            cla_sort_max = class_num_sort_arr.max()
            aj_count = len(class_num_sort_arr)
#             print(cla_sort_max)
            if cla_sort_max < aj_min:
                # 逐个处理每个类的值，最大值小于100，则调整到120
                ratio = aj_min / cla_sort_max
                for lc in range(len(c)):                 
                    if c[lc] in class_num_sort_arr and aj_count > 0:
                        c[lc] = aj_min
                        aj_count -= 1
                        continue
                    elif c[lc] == c.max() and c.max() <100:
                        c[lc] = 120
                        continue
                    elif c[lc] < aj_min and c[lc] !=0 :
                        c[lc] += aj_min
                        
            elif aj_min <= cla_sort_max and len(class_num_sort)!= 1:
                
                for lc in range(len(c)):
                    if c[lc] in class_num_sort_arr and aj_count > 0:
                        c[lc] = aj_min
                        aj_count -= 1

            cla_num_per_client[lenc] = c.tolist()
            print(aj_min)
            print(class_num_sort_arr)
            print(c.tolist())
            print('++++++++++')
#         print(_tmp)
        print("******************************")
        print(cla_num_per_client)

        for i in range(num_users):
            user_info_static[i] = {}
            cla_num = 0
            class_ids = []
            for di in cla_num_per_client[i]:
                if di == 0:
                    cla_num += 1
                    continue
                else:
                    class_ids.append(cla_num)
                cla_num += 1
            user_info_static[i]['class_ids'] = class_ids
#             print(class_ids)
            user_info_static[i]['num_per_class'] = cla_num_per_client[i]

    if train == True:
        user_class_comp = np.array(cla_num_per_client)
        cal_user_weight = get_weight(user_class_comp)
    if train == False:
        for i in range(num_users):
            for num in range(len(user_info_static[i]['num_per_class'])):
                user_info_static[i]['num_per_class'][num] = int(args.tt_ratio * user_info_static[i]['num_per_class'][num] + 1)

    for i in range(num_users):
        rand_set = []
        select_labels = user_info_static[i]['class_ids']
#         print(user_info_static[i]['num_per_class'])
        for j in range(len(select_labels)):
            # number per class
            class_samples_per_user = int(user_info_static[i]['num_per_class'][select_labels[j]])
#             print([select_labels[j]],'-',class_samples_per_user)
#             print(len(idxs_dict[select_labels[j]]),class_samples_per_user )
            if class_samples_per_user == 0:
                print('zero')
                continue
            if class_samples_per_user >= len(idxs_dict[select_labels[j]]):
                class_samples_per_user = len(idxs_dict[select_labels[j]])-1
            idx = np.random.choice(len(idxs_dict[select_labels[j]]),class_samples_per_user ,replace=False)
#             print(i,' idx ',len(idx),'class_samples_per_user ',class_samples_per_user)
            for ids in idx:
                rand_set.append(idxs_dict[select_labels[j]][ids])
#             for k in range(class_samples_per_user):
#                 if len(idxs_dict[select_labels[j]]) > 0:
#                     idx = np.random.choice(len(idxs_dict[select_labels[j]]), replace=False)
#                     rand_set.append(idxs_dict[select_labels[j]][idx])
#                 #                     idxs_dict[select_labels[j]].pop(idx)
#                 else:
#                     idx = np.random.choice(len(idxs_dict_copy[select_labels[j]]), replace=False)
#                     rand_set.append(idxs_dict_copy[select_labels[j]][idx])
#         #                     idxs_dict_copy[select_labels[j]].pop(idx)
#         print(i,' number of samples:',len(rand_set))
        dict_users[i] = np.array(rand_set, dtype='int64')
    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)
    if train == True:
        data_path = 'data/'+str(args.dataset)+'/list_data' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
            args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
            args.min_class) + '_client_' + str(args.num_users) + '.json'         
        with open(data_path, 'w') as outfile:
            json.dump(cla_num_per_client, outfile)
        return dict_users, user_info_static, cal_user_weight
    else:
        return dict_users, user_info_static