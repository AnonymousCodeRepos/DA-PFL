#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 权重随重叠部分的长度而变化

import math
import random
import numpy as np
import torch
import copy

def aj_cos_sim(a, b):
#     a_m = a - np.mean([a[i]+b[i] for i in range (len(a))], axis=0)
#     b_m = b - np.mean([a[i]+b[i] for i in range (len(a))], axis=0)
    a_m = a - np.mean(a+b)
    b_m = b - np.mean(a+b)
    a_norm = np.linalg.norm(a_m)
    b_norm = np.linalg.norm(b_m)
    if a_norm * b_norm == 0:
        return 1
    ajcos = np.dot(a_m, b_m) / (a_norm * b_norm)
    return 2-ajcos

def com(a, b):
    l = len(a)
    tmpc = []
    tmpd = []

    for i in range(0,l):
        if a[i]!=0 and b[i]!=0:
            tmpc.append(a[i])
            tmpd.append(b[i])
#     return a,b
    return tmpc, tmpd

def list_sub(a,b):
    ans = 0.0
    for i in range(len(a)):
        ans += b[i]-a[i]
    
    return ans

def aj_cos_all(lis):
    lenth = len(lis)
    ans = np.zeros((lenth, lenth))
    for i in range(0, lenth):
        li = len(lis[i])
        sumi = sum(lis[i])
        for j in range(0, lenth):
            a,b = com(lis[i],lis[j])
            a_com = len(a)
            a_sum = sum(a)
            ratio = 1.0 - (1.0 * a_sum / sumi)
            if a == [] or b == []:
                ans[i][j] = 0.0
                continue
            elif len(a) == 1:
                ans[i][j] = 0.0
            else:
                ans[i][j] = aj_cos_sim(a, b)*(a_com/li)
#                 print(a,b,ratio,a_com/li, aj_cos_sim(a, b),ans[i][j])
    return ans


def arr_normal(array):
#     array = array.tolist()
    arr_min = np.min(array[np.nonzero(array)])*0.9    
    array[np.where(array == 0.0)] = arr_min 
#     for i in range(len(array)):
#         for j in range(len(array[i])):
#             if array[i][j] == 0.0:
#                 array[i][j] = array[i][i]
                
    for i in range(len(array)):
        sumi = array[i].sum()
        if sumi == 0:
            sumi = 1.0
        for j in range(len(array[i])):
            array[i][j] /= sumi*1.0
 
    
    return array        


def get_weight(nump_arr):
    nump_int = nump_arr.astype(int)
    nums = nump_int.tolist()
    weights = aj_cos_all(nums)
    weights = arr_normal(weights)
    return weights