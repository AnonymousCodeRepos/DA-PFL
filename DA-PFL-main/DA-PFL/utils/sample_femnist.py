# Credit to Tian Li https://github.com/litian96/FedDANE/tree/master/data/nist/data

from __future__ import division
import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
# from data_options import args_parser
from PIL import Image

from utils.weighting import get_weight


# NUM_USER = 200
# CLASS_PER_USER = 2  # from 10 lowercase characters

# num_samples_1 = [90, 70, 30]
# num_samples_n = [30, 70, 90]
# tt_rate = 0.8

def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90:  # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61)  # lowercase


def load_image(file_name):
    '''read in a png
    Return: a flatted list representing the image
    '''
    size = (28, 28)
    img = Image.open(file_name)
    gray = img.convert('L')
    gray.thumbnail(size, Image.ANTIALIAS)
    arr = np.asarray(gray).copy()
    vec = arr.flatten()
    vec = vec / 255  # scale all pixel values to between 0 and 1
    vec = vec.tolist()

    return vec


def sampling_femnist(args):
    file_dir = "leaf-master/data/femnist/data/raw_data/by_class"
    NUM_USER = args.num_users

    train_path = 'data/femnist/femnist_train' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
        args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
        args.min_class) + '_client_' + str(args.num_users) + '.json'
    test_path = 'data/femnist/femnist_test' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
        args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
        args.min_class) + '_client_' + str(args.num_users) + '.json'
    weight_path = 'data/femnist/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
        args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
        args.min_class) + '_client_' + str(args.num_users) + '.json'
    data_path = 'data/femnist/list_data' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(
        args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
        args.min_class) + '_client_' + str(args.num_users) + '.json'    

    X = [[] for _ in range(NUM_USER)]
    y = [[] for _ in range(NUM_USER)]
    Xt = [[] for _ in range(NUM_USER)]
    yt = [[] for _ in range(NUM_USER)]
    nist_data = {}

    cal_user_weight = np.zeros((args.num_users, args.num_users))
    user_class_comp = np.zeros((args.num_users, args.num_classes))

    for class_ in os.listdir(file_dir):

        real_class = relabel_class(class_)
        if real_class >= 36 and real_class <= 45:
            full_img_path = file_dir + "/" + class_ + "/train_" + class_
            all_files_this_class = os.listdir(full_img_path)
            random.shuffle(all_files_this_class)
            sampled_files_this_class = all_files_this_class[:4000]
            imgs = []
            for img in sampled_files_this_class:
                imgs.append(load_image(full_img_path + "/" + img))
            class_ = relabel_class(class_)
            print(class_)
            nist_data[class_ - 36] = imgs  # a list of list, key is (0, 25)
            print(len(imgs))

    idx = np.zeros(10, dtype=np.int64)

    diri_alpha_list = []
    diri_alpha_list.append(args.diri_alpha)

    # generate dirichlet distribution for all clients
    # k*N, class_number * clients
    dirichlet = np.random.dirichlet(diri_alpha_list * args.num_users, args.num_classes)
    # generate sample numbers for every classes
    if args.num_users <100:
        if args.diri_alpha < 1.0:
            dirichlet = dirichlet * (args.num_users * 50)
        else:
            dirichlet = dirichlet * (args.num_users * 50)
    else:
        dirichlet = dirichlet * (args.num_users * 60)
    # N * k: ,but some value need to handle
    _tmp = dirichlet.T
    # N * k:
    cla_num_per_client = []
    for t in _tmp:
        t = [int(tt) for tt in t]
        cla_num_per_client.append(t)

    for user in range(NUM_USER):
        _class_client = np.array(cla_num_per_client[user])
        # set the minimum number of majority class to 120
        if _class_client.max() < 120:
            _idx = np.argwhere(_class_client == _class_client.max()).flatten()
            _class_client[_idx[0]] = 120

        aj_min = int(_class_client.max() / args.imbalance_ratio)

        if _class_client.min() >= aj_min:
            _idx = np.argwhere(_class_client == _class_client.min()).flatten()
            _class_client[_idx[0]] = aj_min
        else:
            _idx = np.argwhere(_class_client == _class_client.min()).flatten()
            _class_client[_idx[0]] = aj_min
        class_id = 0
        for j in range(len(_class_client)):
            if _class_client[j] == 0:
                class_id += 1
                continue
            else:
                if _class_client[j] < aj_min:
                    _class_client[j] += aj_min

            if idx[class_id] + _class_client[j] > len(nist_data[class_id]):
                idx[class_id] = 0
            train_l = int((1 - args.tt_ratio) * _class_client[j])
            test_l = _class_client[j] - train_l
            X[user] += nist_data[class_id][idx[class_id]: (idx[class_id] + train_l)]
            y[user] += (class_id * np.ones(train_l)).tolist()
            
            Xt[user] += nist_data[class_id][(idx[class_id] + train_l): (idx[class_id] + train_l+ test_l)]
            yt[user] += (class_id * np.ones(test_l)).tolist()
            idx[class_id] += _class_client[j]
            class_id += 1
#             print('user', user, 'train_l :', train_l, 'test_l', test_l)
        print('user', user, 'X[user] :', len(X[user]), 'Xt[user]', len(Xt[user]))
        cla_num_per_client[user] = _class_client.tolist()
    print(cla_num_per_client)

    user_class_comp = np.array(cla_num_per_client)
    cal_user_weight = get_weight(user_class_comp)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:05d}'.format(i)

        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        
        combinedt = list(zip(Xt[i], yt[i]))
        random.shuffle(combinedt)
        Xt[i][:], yt[i][:] = zip(*combinedt)
        
        num_samples = len(X[i])
        train_len = len(X[i])
        test_len = len(Xt[i])

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i], 'y': y[i]}
        train_data['num_samples'].append(train_len)
        
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': Xt[i], 'y': yt[i]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)
    with open(weight_path, 'w') as outfile:
        json.dump(cal_user_weight.tolist(), outfile)
    with open(data_path, 'w') as outfile:
        json.dump(cla_num_per_client, outfile)



