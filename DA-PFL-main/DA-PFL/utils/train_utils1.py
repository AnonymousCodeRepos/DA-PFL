# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang

from torchvision import datasets, transforms
from models.Nets import CNNCifar, CNNCifar100, RNNSent, MLP,CNN_FEMNIST, AlexNet
from utils.sampling_cifar10 import new_noniid
import os
import json
import numpy as np

from utils.MyEncoder import MyEncoder

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])



trans_imagenet = transforms.Compose([ transforms.RandomResizedCrop(224), 
                                     transforms.RandomHorizontalFlip(), 
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


trans_imagenet_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trans_imagenet_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        dict_users_train, rand_set_all, weight= new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args)
        dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args, user_info_static=rand_set_all)
        # sample users
#         dict_users_train, rand_set_all = new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
#         dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all, weight= new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args)
        dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args, user_info_static=rand_set_all)
    elif args.dataset == 'cifar100':#这里还没检查，只是改了函数名
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dict_users_train, rand_set_all, weight = new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args)
        dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args, user_info_static=rand_set_all)
    elif args.dataset == 'imagenet':
        dataset_train = datasets.ImageNet('data/imagenet', split= 'train', transform=trans_imagenet)#, transform=trans_cifar100_train
        dataset_test = datasets.ImageNet('data/imagenet', split= 'train', transform=trans_imagenet)#, transform=trans_cifar100_val
        dict_users_train, rand_set_all, weight = new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args)
        dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args, user_info_static=rand_set_all)
        
        
    else:
        exit('Error: unrecognized dataset')
        
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'mnist'or args.dataset == 'imagenet':
        return dataset_train, dataset_test, dict_users_train, dict_users_test, weight
    else:
        return dataset_train, dataset_test, dict_users_train, dict_users_test

def read_data(train_data_dir,train_json, test_data_dir,test_json):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

#     train_files = os.listdir(train_data_dir)
#     train_files = [f for f in train_files if f.endswith('.json')]
    train_files = train_json
    file_path = os.path.join(train_data_dir,train_files)
    with open(file_path, 'r') as inf:
        cdata = json.load(inf)
    clients.extend(cdata['users'])
    if 'hierarchies' in cdata:
        groups.extend(cdata['hierarchies'])
    train_data.update(cdata['user_data'])
#     for f in train_files:
#         file_path = os.path.join(train_data_dir,f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         clients.extend(cdata['users'])
#         if 'hierarchies' in cdata:
#             groups.extend(cdata['hierarchies'])
#         train_data.update(cdata['user_data'])

#     test_files = os.listdir(test_data_dir)
#     test_files = [f for f in test_files if f.endswith('.json')]
    test_files = test_json
    file_path = os.path.join(test_data_dir,test_files)
    with open(file_path, 'r') as inf:
        cdata = json.load(inf)
    test_data.update(cdata['user_data'])
#     for f in test_files:
#         file_path = os.path.join(test_data_dir,f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_model(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    elif 'sent140' in args.dataset:
        net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
    elif args.model == 'alexnet' and 'imagenet' in args.dataset:
        net_glob = AlexNet(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob


# def save_data_cifar(dataset_train, dataset_test, dict_users_train, dict_users_test, weight,
#               train_json,test_json,dic_train_json, dic_test_json,weight_json):
def save_data_cifar(dict_users_train, dict_users_test, weight, dic_train_json, dic_test_json,weight_json):
#     with open(train_json, 'w') as outfile:
#         json.dump(dataset_train, outfile)
#     with open(test_json, 'w') as outfile:
#         json.dump(dataset_test, outfile)

# array type cannot save to json file
# it needs transform to list
    save_dict_users_train = {}
    save_dic_test_json = {}
    save_weight = weight.tolist()
    
    for i in dict_users_train.keys():
        save_dict_users_train[i] = dict_users_train[i].tolist()
    for i in dict_users_test.keys():
        save_dic_test_json[i] = dict_users_test[i].tolist()
        
    with open(dic_train_json, 'w') as outfile:
        json.dump(save_dict_users_train, outfile)
    with open(dic_test_json, 'w') as outfile:
        json.dump(save_dic_test_json, outfile)
        
    with open(weight_json, 'w') as outfile:
        json.dump(save_weight, outfile)
        
        
def load_data_cifar(args, dic_train_json, dic_test_json, weight_json):
#     dataset_train = {}
#     dataset_test = {}
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
    elif args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
#         # sample users
#         dict_users_train, rand_set_all = new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
#         dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar100':# 
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
#         dict_users_train, rand_set_all = new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
#         dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    elif args.dataset == 'imagenet':
        dataset_train = datasets.ImageNet('data/imagenet', split= 'train', transform=trans_imagenet)
        dataset_test = datasets.ImageNet('data/imagenet', split= 'train', transform=trans_imagenet)
#         dict_users_train, rand_set_all, weight = new_noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args)
#         dict_users_test, rand_set_all = new_noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args, user_info_static=rand_set_all)
    else:
        exit('Error: unrecognized dataset')
        
    dict_users_train = {}
    dict_users_test = {}
    _dict_users_train = {}
    _dict_users_test = {}
    weight = np.zeros((args.num_users, args.num_users))
    
#     with open(train_json, 'r') as file:
#         dataset_train = json.load(file)    
    
#     with open(test_json, 'r') as file:
#         dataset_test = json.load(file)
        
    with open(dic_train_json, 'r') as file:
        _dict_users_train = json.load(file)  
        
    with open(dic_test_json, 'r') as file:
        _dict_users_test = json.load(file)
        
    with open(weight_json, 'r') as file:
        _weight = json.load(file)
        
    for i in _dict_users_train.keys():
        dict_users_train[i] = np.array(_dict_users_train[i])
        dict_users_train[int(i)] = dict_users_train.pop(i)
        
    for i in _dict_users_test.keys():
        dict_users_test[i] = np.array(_dict_users_test[i])
        dict_users_test[int(i)] = dict_users_test.pop(i)
        
    weight = np.array(_weight)
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet':
        return dataset_train, dataset_test, dict_users_train, dict_users_test, weight
    else:
        return dataset_train, dataset_test, dict_users_train, dict_users_test
    
def load_data_femnist(args):
    if args.dataset == 'femnist':
        weight_json = 'data/femnist/weight' + '_alpha_' + str(args.diri_alpha) + '_IMratio_' + str(args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(args.min_class) + '_client_' + str(args.num_users) + '.json' 
        
        with open(weight_json, 'r') as file:
            _weight = json.load(file)
            
        weight = np.array(_weight)
        return weight 
    else:
        print('wrong dataset! (from --> train_utils.py)')
    
def load_data_compose(args):
    list_original_data_per_client_json = 'data/'+ str(args.dataset) +'/list_data' + '_alpha_' + str(
        args.diri_alpha) + '_IMratio_' + str(
        args.imbalance_ratio) + '_testset_' + str(args.tt_ratio) + '_mino_classes_' + str(
        args.min_class) + '_client_' + str(args.num_users) + '.json'#../FedRep1/FedRep/
    with open(list_original_data_per_client_json, 'r') as file:
        list_original_data_per_client = json.load(file)
        
    return list_original_data_per_client

