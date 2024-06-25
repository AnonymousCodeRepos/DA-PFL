# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
# import FedProx
from torch.optim import lr_scheduler

from models.test import test_img_local
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y


def get_wei(model):
    pa = []
    for key in model.keys():
        pa.append(model[key].view(-1))
    pa = torch.cat(pa)
    return pa


# data_sets[p] = datasets.ImageFolder(data_dirs[p], transform=data_transforms[p],)

# data_loaders[p] = DataLoader(data_sets[p], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, sampler=None )


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

# Generic local update class, implements local updates for FedRep, FedAvg, FedProx
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        bias_p = []
        weight_p = []

        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        #         scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                                        lr=lr,
                                        gmf=self.args.gmf,
                                        mu=self.args.mu,
                                        ratio=1 / self.args.num_users,
                                        momentum=0.5,
                                        nesterov=False,
                                        weight_decay=1e-4)

        local_eps = self.args.local_ep
        if last:
            if self.args.alg == 'fedavg' or self.args.alg == 'prox':
                local_eps = 10
                net_keys = [*net.state_dict().keys()]
                if 'cifar' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 3, 4]]
                elif 'sent140' in self.args.dataset:
                    w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
                elif 'mnist' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 2]]
                elif 'imagenet' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [4, 5, 6, 7, 8] ] #[1, 2, 3]
            elif 'maml' in self.args.alg:
                local_eps = 5
                w_glob_keys = []
            else:
                local_eps = max(10, local_eps - self.args.local_rep_ep)

        head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif iter == head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            # learning rate
            #             scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd


class LocalUpdateDAPFL(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if indd is not None:
            self.indd = indd
        else:
            self.indd = None

    def train(self, net, ind=None, agg_w=None, lam=1, idx=-1, lr=0.1, last=False, we=0.0):
        net.train()
        # train and update
        bias_p = []
        weight_p = []
        w_glob_keys = []
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        local_eps = self.args.local_ep
        args = self.args
        epoch_loss = []
        num_updates = 0
        for iter in range(local_eps):
            done = False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                w_0 = copy.deepcopy(net.state_dict())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                net_tmp = copy.deepcopy(net)
                net_tmp.load_state_dict(agg_w)
                mse_loss = nn.MSELoss(reduction='sum')
                loss.backward()
                optimizer.step()

                if agg_w is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - args.lr * lam * (w_0[key] - agg_w[key])
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()


                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates >= self.args.local_updates:
                    done = True
                    break
            #             learning rate
            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

