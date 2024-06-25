# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
# modified
from utils.train_utils1 import get_data, get_model, read_data, save_data_cifar, load_data_cifar, load_data_femnist
from models.Update import LocalUpdate, LocalUpdateDAPFL
from models.test import test_img_local_all
from utils.affinity_model import get_aff_model
import pdb
import time
import sys
import random
import os
from utils.sample_femnist import sampling_femnist


if __name__ == '__main__':
    # parse args

    args = args_parser()
    torch.set_num_threads(1)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    exp_count = 0
    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        if args.dataset == 'cifar10':
            if args.gen_data == 1:
                dataset_train, dataset_test, dict_users_train, dict_users_test, weight = get_data(args)

                dic_train_json = 'data/cifar10/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dic_test_json = 'data/cifar10/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                weight_json = 'data/cifar10/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                for idx in dict_users_train.keys():
                    np.random.shuffle(dict_users_train[idx])
                save_data_cifar(dict_users_train, dict_users_test, weight, dic_train_json, dic_test_json, weight_json)

            else:
                dic_train_json = 'data/cifar10/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dic_test_json = 'data/cifar10/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                weight_json = 'data/cifar10/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dataset_train, dataset_test, dict_users_train, dict_users_test, weight = load_data_cifar(args,
                                                                                                         dic_train_json,
                                                                                                         dic_test_json,
                                                                                                         weight_json)
        elif args.dataset == 'mnist':
            if args.gen_data == 1:
                dataset_train, dataset_test, dict_users_train, dict_users_test, weight = get_data(args)
                dic_train_json = 'data/mnist/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dic_test_json = 'data/mnist/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                weight_json = 'data/mnist/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                for idx in dict_users_train.keys():
                    np.random.shuffle(dict_users_train[idx])
                save_data_cifar(dict_users_train, dict_users_test, weight, dic_train_json, dic_test_json, weight_json)

            else:
                dic_train_json = 'data/mnist/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dic_test_json = 'data/mnist/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                weight_json = 'data/mnist/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dataset_train, dataset_test, dict_users_train, dict_users_test, weight = load_data_cifar(args,
                                                                                                         dic_train_json,
                                                                                                         dic_test_json,
                                                                                                         weight_json)

        if args.dataset == 'cifar100':
            if args.gen_data == 1:
                dataset_train, dataset_test, dict_users_train, dict_users_test, weight = get_data(args)
                #             train_json = 'data/cifar10/train.json'
                #             test_json = 'data/cifar10/test.json'
                dic_train_json = 'data/cifar100/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dic_test_json = 'data/cifar100/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                weight_json = 'data/cifar100/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                for idx in dict_users_train.keys():
                    np.random.shuffle(dict_users_train[idx])
                save_data_cifar(dict_users_train, dict_users_test, weight, dic_train_json, dic_test_json, weight_json)

            else:
                dic_train_json = 'data/cifar100/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dic_test_json = 'data/cifar100/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                weight_json = 'data/cifar100/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                    args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                    args.min_class) + '_client_' + str(args.num_users) + '.json'
                dataset_train, dataset_test, dict_users_train, dict_users_test, weight = load_data_cifar(args,
                                                                                                         dic_train_json,
                                                                                                         dic_test_json,
                                                                                                         weight_json)
    elif 'imagenet' in args.dataset:
        if args.gen_data == 1:
            dataset_train, dataset_test, dict_users_train, dict_users_test, weight = get_data(args)

            dic_train_json = 'data/imagenet/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                args.min_class) + '_client_' + str(args.num_users) + '.json'
            dic_test_json = 'data/imagenet/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                args.min_class) + '_client_' + str(args.num_users) + '.json'
            weight_json = 'data/imagenet/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                args.min_class) + '_client_' + str(args.num_users) + '.json'
            for idx in dict_users_train.keys():
                np.random.shuffle(dict_users_train[idx])
            save_data_cifar(dict_users_train, dict_users_test, weight, dic_train_json, dic_test_json, weight_json)
        else:
            dic_train_json = 'data/imagenet/dic_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                args.min_class) + '_client_' + str(args.num_users) + '.json'
            dic_test_json = 'data/imagenet/dic_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                args.min_class) + '_client_' + str(args.num_users) + '.json'
            weight_json = 'data/imagenet/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
                args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
                args.min_class) + '_client_' + str(args.num_users) + '.json'
            dataset_train, dataset_test, dict_users_train, dict_users_test, weight = load_data_cifar(args,
                                                                                                     dic_train_json,
                                                                                                     dic_test_json,
                                                                                                     weight_json)
    else:
        if args.gen_data == 1:
            if 'femnist' in args.dataset:
                sampling_femnist(args)
            else:
                train_path = './leaf-master/data/' + args.dataset + '/data/train'
                test_path = './leaf-master/data/' + args.dataset + '/data/test'
        train_path = 'data/femnist'
        test_path = 'data/femnist'
        train_json = 'femnist_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
            args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
            args.min_class) + '_client_' + str(args.num_users) + '.json'
        test_json = 'femnist_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
            args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
            args.min_class) + '_client_' + str(args.num_users) + '.json'
        clients, groups, dataset_train, dataset_test = read_data(train_path, train_json, test_path, test_json)
        weight = load_data_femnist(args)

        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    net_glob = get_model(args)
    net_glob.train()

    w_glob_keys = []

    # first running, this
    # init list of local models for each user
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
    net_local = copy.deepcopy(net_glob)

    p_globals = {}
    for user in range(args.num_users):
        p_dict = {}
        for key in net_glob.state_dict().keys():
            p_dict[key] = net_glob.state_dict()[key]
        p_globals[user] = p_dict
    net_p = copy.deepcopy(net_glob)

    # training
    loss_train = []
    test_freq = args.test_freq
    indd = None
    accs = []
    accs10 = 0
    times = []
    times_in = []
    lam = args.lam_ditto
    start = time.time()
    stime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))
    print('start time: ', stime_str)
    acc_all = []
    loss_all = []
    for iter in range(args.epochs + 1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)

        # last epoch:
        if iter == args.epochs:
            m = args.num_users

        # select some clients
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # calculate aff models
        p_globals = copy.deepcopy(
            get_aff_model(args=args, w_locals=w_locals, p_globals=copy.deepcopy(p_globals), idxs_users=idxs_users,
                          weight=weight))

        # selected clients training:
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            if 'femnist' in args.dataset:
                if iter == args.epochs:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:]],
                                        idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdateDAPFL(args=args,
                                             dataset=dataset_train[list(dataset_train.keys())[idx][:]],
                                             idxs=dict_users_train, indd=indd)
            else:  #
                if iter == args.epochs:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:])
                else:

                    local = LocalUpdateDAPFL(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:])

            net_global = copy.deepcopy(net_glob)
            w_glob_k = copy.deepcopy(net_global.state_dict())

            w_local = copy.deepcopy(w_locals[idx])
            net_local.load_state_dict(w_local)

            net_p.load_state_dict(p_globals[idx])

            p_glob_k = copy.deepcopy(net_p.state_dict())
            # local model training process
            if 'femnist' in args.dataset:
                if iter != args.epochs:
                    w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx],
                                                      lr=args.lr,
                                                      agg_w=p_glob_k, lam=lam, we=0.0)
                else:
                    last = iter == args.epochs
                    w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx],
                                                      w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            else:
                if iter != args.epochs:
                    w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, lr=args.lr,
                                                      agg_w=p_glob_k, lam=lam)

            loss_locals.append(copy.deepcopy(loss))
            w_k = w_local
            if len(w_glob) == 0:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_k[key] / m

                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_k[key] / m
                    w_locals[idx][key] = w_local[key]

            times_in.append(time.time() - start_in)
        p_globals = copy.deepcopy(get_aff_model(args=args, w_locals=w_locals, p_globals=copy.deepcopy(p_globals), idxs_users=idxs_users, weight=weight))

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        # get weighted average for global weights
        # A round of training has been completed, global model updates.
        net_glob.load_state_dict(w_glob)

        if times == []:
            times.append(max(times_in))
        else:
            times.append(times[-1] + max(times_in))
        acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test, w_locals=w_locals,
                                                 indd=indd, dataset_train=dataset_train,
                                                 dict_users_train=dict_users_train,
                                                 return_all=False)
        accs.append(acc_test)
        acc_all.append(acc_test)
        loss_all.append(loss_test)
        if iter != args.epochs:
            print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

        else:
            print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                loss_avg, loss_test, acc_test))
        if iter >= args.epochs - 10 and iter != args.epochs:
            accs10 += acc_test / 10

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    end = time.time()
    etime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))
    print('end time: ', etime_str)
    print(end - start)
    # print(times)
    print("accs", accs)
    print('exp0', exp_count)

    print("all loss_train train set:", loss_train)

    print("all acc_all test set:", acc_all)
    print("all loss_all test set:", loss_all)

