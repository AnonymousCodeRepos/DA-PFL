
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn



def get_aff_model(args=None, w_locals=None, p_globals=None, idxs_users=None, weight=None):
    w_c = np.zeros((args.num_users, args.num_users))
    for i in range(0, args.num_users):
        tmp = []
        pt = 0

        for j in range(0, args.num_users):
            e_wei = []
            exp_wei = 1.0

            if i in idxs_users and j in idxs_users:
                pt += 1
                for k1 in w_locals[i].keys():

                    w_lay = torch.norm(torch.sub(w_locals[i][k1], w_locals[j][k1])).float()
                    e_wei.append(w_lay)

                exp_wei = 1.0 - float(torch.exp(torch.mul(-1.0, torch.norm(torch.Tensor(e_wei)))).float())
                # for init model:
                if exp_wei == 0.0:
                    exp_wei = 1.0

            if i in idxs_users:
                if j in idxs_users:
                    w_c[i][j] = weight[i][j] * exp_wei
                    tmp.append(w_c[i][j])
            else:
                w_c[i][j] = weight[i][j]

        t_sum = sum(tmp)

        if i in idxs_users:
            if t_sum == 0:
                t_sum = 1

            for a in idxs_users:
                w_c[i][a] /= t_sum * 1.0

    for i in range(0, args.num_users):
        for j in range(0, args.num_users):
            for key in w_locals[j].keys():
                # first re-weight :
                if j == 0:
                    p_globals[i][key] = torch.mul(w_c[i][j], w_locals[j][key])
                else:
                    p_globals[i][key] = torch.add(p_globals[i][key], torch.mul(w_c[i][j], w_locals[j][key]))

    return p_globals