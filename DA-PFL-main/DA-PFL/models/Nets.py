# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import json
import numpy as np
from models.language_utils import get_word_emb_arr

# class MLP(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(MLP, self).__init__()
#         self.layer_input = nn.Linear(dim_in, 512)
        
#         self.relu = nn.ReLU()
# #         self.dropout1 = nn.Dropout(p=0.5)
# #         self.layer_hidden1 = nn.Linear(512, 256)
# #         self.dropout2 = nn.Dropout(p=0.5)
#         self.layer_hidden2 = nn.Linear(512, 128)
# #         self.dropout3 = nn.Dropout(p=0.5)
#         self.layer_out = nn.Linear(128, dim_out)
# #         self.dropout4 = nn.Dropout(p=0.6)
#         self.softmax = nn.Softmax(dim=1)
#         self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
# #                             ['layer_hidden1.weight', 'layer_hidden1.bias'],
#                             ['layer_hidden2.weight', 'layer_hidden2.bias'],
#                             ['layer_out.weight', 'layer_out.bias']
                            
#                             ]

#     def forward(self, x):
#         x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#         x = self.layer_input(x)
# #         x = self.dropout1(x)
#         x = self.relu(x)
# #         x = self.dropout1(x)#新增
# #         x = self.layer_hidden1(x)
# #         x = self.dropout2(x)
# #         x = self.relu(x)
#         x = self.layer_hidden2(x)
# #         x = self.dropout3(x)
#         x = self.relu(x)
#         x = self.layer_out(x)
# #         x = self.dropout4(x)
#         return self.softmax(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.dropout1 = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
#         
        self.layer_hidden1 = nn.Linear(512, 256)
#         self.dropout2 = nn.Dropout(p=0.5)
        self.layer_hidden2 = nn.Linear(256, 64)
#         self.dropout3 = nn.Dropout(p=0.5)
        self.layer_out = nn.Linear(64, dim_out)
#         self.dropout4 = nn.Dropout(p=0.6)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout1(x)
        x = self.relu(x)
#        #新增
        x = self.layer_hidden1(x)
#         x = self.dropout2(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
#         x = self.dropout3(x)
        x = self.relu(x)
        x = self.layer_out(x)
#         x = self.dropout4(x)
        return self.softmax(x)

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, args.num_classes)
        self.cls = args.num_classes
        self.drop = nn.Dropout(0.6)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(self.fc3(x))
        return F.log_softmax(x, dim=1)

class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)
        self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNN_FEMNIST(nn.Module):
    def __init__(self, args):
        super(CNN_FEMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class RNNSent(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by: Hongyi Wang from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self,args, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, emb_arr=None):
        super(RNNSent, self).__init__()
        VOCAB_DIR = 'models/embs.json'
        emb, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.encoder = torch.tensor(emb).to(args.device)
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.fc = nn.Linear(nhid, 10)
        self.decoder = nn.Linear(10, ntoken)

        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.device = args.device

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        input = torch.transpose(input, 0,1)
        emb = torch.zeros((25,4,300)) 
        for i in range(25):
            for j in range(4):
                emb[i,j,:] = self.encoder[input[i,j],:]
        emb = emb.to(self.device)
        emb = emb.view(300,4,25)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(F.relu(self.fc(output)))
        decoded = self.decoder(output[-1,:,:])
        return decoded.t(), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        

# class AlexNet(nn.Module):
#     def __init__(self, args):
#         super(AlexNet, self).__init__()     # nn.Module子类的函数必须在构造函数中执行父类的构造函数，这个是父类的构造函数
#         # 5个卷积层
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             # inplace=True
#             # 计算结果不会有影响。利用inplace计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。
#             # 但是会对原变量覆盖，只要不带来错误就用。在这里个人理解其意思就是relu后会覆盖原来的卷积后的值，减少存储量，
#             # 毕竟也用不上了，能不要则不要
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         # 三个全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, args.num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.fc(x)
# #         return x
#         return F.log_softmax(x, dim=1)
        
        


# class AlexNet(nn.Module):
#     def __init__(self, args):
#         super(AlexNet, self).__init__()

#         self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 11, 4, 2),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2, 2), )

#         self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 5, 1, 2),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2, 2),  )

#         self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1, 1),
#                                    nn.ReLU(),
#                                    nn.Conv2d(384, 384, 3, 1, 1),
#                                    nn.ReLU(),
#                                    nn.Conv2d(384, 256, 3, 1, 1),
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2, 2))


#         self.fc=nn.Sequential(nn.Linear(256*6*6, 4096),
#                                 nn.ReLU(),
#                                 nn.Dropout(0.5),
#                                 nn.Linear(4096, 4096),
#                                 nn.ReLU(),
#                                 nn.Dropout(0.5),
#                                 nn.Linear(4096, args.num_classes),
#                                 )
# #         self.cls = args.num_classes
#         self.weight_keys = [['fc.weight', 'fc.bias'],
#                             ['conv3.weight', 'conv3.bias'],
#                             ['conv2.weight', 'conv2.bias'],
#                             ['conv1.weight', 'conv1.bias'],]

#     def forward(self, x):
#         x=self.conv1(x)
#         x=self.conv2(x)
#         x=self.conv3(x)
#         output=self.fc(x.view(-1, 256*6*6))
# #         return output
#         return F.log_softmax(output, dim=1)

class AlexNet(nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()
        # 卷积层，输入大小为224*224，输出大小为55*55，输入通道为3，输出为96，卷积核为11，步长为4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # 使用ReLU作为激活函数
        self.ReLU = nn.ReLU()
        # MaxPool2d：最大池化操作
        # 最大池化层，输入大小为55*55，输出大小为27*27，输入通道为96，输出为96，池化核为3，步长为2
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为27*27，输出大小为27*27，输入通道为96，输出为256，卷积核为5，扩充边缘为2，步长为1
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 最大池化层，输入大小为27*27，输出大小为13*13，输入通道为256，输出为256，池化核为3，步长为2
#         self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为256，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为256，卷积核为3，扩充边缘为1，步长为1
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 最大池化层，输入大小为13*13，输出大小为6*6，输入通道为256，输出为256，池化核为3，步长为2
#         self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层
        # Linear（in_features，out_features）
        # in_features指的是[batch_size, size]中的size,即样本的大小
        # out_features指的是[batch_size，output_size]中的output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.fc6 = nn.Linear(6*6*256, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        # 全连接层&softmax
        self.fc8 = nn.Linear(4096, 1000)
        self.fc9 = nn.Linear(1000, args.num_classes)
        self.weight_keys = [['fc9.weight', 'fc9.bias'],['fc8.weight', 'fc8.bias'],['fc7.weight', 'fc7.bias'],['fc6.weight', 'fc6.bias'],
                            ['conv5.weight', 'conv5.bias'],
                            ['conv4.weight', 'conv4.bias'],
                            ['conv3.weight', 'conv3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],]
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.pool(x)
        x = self.ReLU(self.conv2(x))
        x = self.pool(x)
        x = self.ReLU(self.conv3(x))
        x = self.ReLU(self.conv4(x))
        x = self.ReLU(self.conv5(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc6(x)
         # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
        # “失活的”神经元不再进行前向传播并且不参与反向传播，这个技术减少了复杂的神经元之间的相互影响
        x = F.dropout(x, p=0.5)
        x = self.fc7(x)
        x = F.dropout(x, p=0.5)
        x = self.fc8(x)
        x = F.dropout(x, p=0.5)
        x = self.fc9(x)
        return F.log_softmax(x, dim=1)



# class AlexNet(nn.Module):
#     def __init__(self, args):
#         super(AlexNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 96, 11, 4, 2) 

#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(96, 256, 5, 1, 2) 


#         self.conv3 = nn.Conv2d(256, 384, 3, 1, 1) 

#         self.conv4 = nn.Conv2d(384, 384, 3, 1, 1),
 
#         self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
 

#         self.fc1= nn.Linear(256*6*6, 4096) 
 
#         self.drop = nn.Dropout(0.5) 
#         self.fc2= nn.Linear(4096, 4096),
 
#         self.fc3= nn.Linear(4096, args.num_classes),
#         self.weight_keys = [['fc1.weight', 'fc1.bias'],['fc2.weight', 'fc2.bias'],['fc3.weight', 'fc3.bias'],['conv5.weight', 'conv5.bias'],
#                             ['conv4.weight', 'conv4.bias'],
#                             ['conv3.weight', 'conv3.bias'],
#                             ['conv2.weight', 'conv2.bias'],
#                             ['conv1.weight', 'conv1.bias'],]

#     def forward(self, x):
#         x=self.conv1(x)
#         x=self.pool(F.relu(self.conv1(x)))
#         x=self.conv2(x)
#         x=self.pool(F.relu(self.conv2(x)))
#         x=self.conv3(x)
#         x=F.relu(self.conv3(x))
#         x=self.conv4(x)
#         x=F.relu(self.conv4(x))
#         x=self.conv5(x)
#         x=self.pool(F.relu(self.conv5(x)))
#         x = x.view(-1, 256*6*6)
#         x = self.drop((F.relu(self.fc1(x))))
#         x = self.drop((F.relu(self.fc2(x))))
#         x = self.fc3(x)
         
# #         return output
#         return F.log_softmax(output, dim=1)


#--------------ResNet------------------

# __all__ = [
#     'ResNet',
#     'resnet18',
#     'resnet34_half',
#     'resnet34',
#     'resnet50_half',
#     'resnet50',
#     'resnet101',
#     'resnet152',
#     'resnext50_32x4d',
#     'resnext101_32x8d',
#     'wide_resnet50_2',
#     'wide_resnet101_2',
# ]

# model_urls = {
#     'resnet18':
#     '{}/resnet/resnet18-epoch100-acc70.316.pth'.format(pretrained_models_path),
#     'resnet34_half':
#     '{}/resnet/resnet34_half-epoch100-acc67.472.pth'.format(
#         pretrained_models_path),
#     'resnet34':
#     '{}/resnet/resnet34-epoch100-acc73.736.pth'.format(pretrained_models_path),
#     'resnet50_half':
#     '{}/resnet/resnet50_half-epoch100-acc72.066.pth'.format(
#         pretrained_models_path),
#     'resnet50':
#     '{}/resnet/resnet50-epoch100-acc76.512.pth'.format(pretrained_models_path),
#     'resnet101':
#     '{}/resnet/resnet101-epoch100-acc77.724.pth'.format(
#         pretrained_models_path),
#     'resnet152':
#     '{}/resnet/resnet152-epoch100-acc78.564.pth'.format(
#         pretrained_models_path),
#     'resnext50_32x4d':
#     'empty',
#     'resnext101_32x8d':
#     'empty',
#     'wide_resnet50_2':
#     'empty',
#     'wide_resnet101_2':
#     'empty',
# }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 inplanes=64,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.interplanes = [
            self.inplanes, self.inplanes * 2, self.inplanes * 4,
            self.inplanes * 8
        ]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.interplanes[0], layers[0])
        self.layer2 = self._make_layer(block,
                                       self.interplanes[1],
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       self.interplanes[2],
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       self.interplanes[3],
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.interplanes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')))

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34_half(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['inplanes'] = 32
    return _resnet('resnet34_half', BasicBlock, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_half(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['inplanes'] = 32
    return _resnet('resnet50_half', Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained,
                   progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)