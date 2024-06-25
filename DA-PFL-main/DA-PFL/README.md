# DA-PFL: Dynamic Affinity Aggregation for Personalized Federated Learning

Authors: Xu Yang, Jiyuan Feng, Songyue Guo, Yongxin Tong, Binxing Fang, and Qing Liao



## Dependencies

Build envoriment:

conda create --name PFL python=3.7 numpy

conda activate PFL

pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102

conda install pandas matplotlib scipy


## Data

This code uses the CIFAR10, CIFAR100 and Federated Extended MNIST (FEMNIST) datasets.

The CIFAR10 and CIFAR100 datasets are downloaded automatically by the torchvision package. 
FEMNIST is provided by the LEAF repository, which should be downloaded from https://github.com/TalwalkarLab/leaf/ and placed in `DAPFL/`. 
Then the raw FEMNIST datasets can be downloaded by following the instructions in LEAF. 
In order to generate the versions of these datasets that we use the paper, we use the following commands from within `DAPFL/leaf-master/data/femnist/`:

FEMNIST: `./preprocess.sh -s niid --sf 0.5 -k 50 -tf 0.8 -t sample`

The FEMNIST must be downloaded through LEAF https://github.com/TalwalkarLab/leaf.

For FEMNIST, we re-sample and re-partition the data to increase its heterogeneity. In order to do so, first navigate to `FedRep/`, then execute 

`mv my_sample.py leaf-master/data/femnist/data/`

`cd leaf-master/data/femnist/data/`

`python my_sample.py`

## Usage

DA-PFL is run using a command of the following form:

`python3 -u main_dapfl.py --dataset [dataset] --model [model] --num_classes [num_classes]  --epochs [epochs] --lr [lr]  --num_users [num_users] --gpu [id]  --local_ep [local_ep] --frac [frac] --local_bs [local_bs] --gen_data [gen_data] --diri_alpha [a] --tt_ratio [ratio] `

Explanation of parameters:

- `dataset` : dataset, may be `cifar10`, `cifar100` and `femnist`
- `num_users` : number of users
- `model` : for the CIFAR datasets, we use `cnn`, for the MNIST datasets, we use `mlp`
- `num_classes` : total number of classes
- `frac` : fraction of participating users in each round 
- `local_bs` : batch size used locally by each user
- `lr` : learning rate
- `epochs` : total number of communication rounds
- `local_ep` : total number of local epochs
- `gpu` : GPU ID
- `gen_data`: generate data (1), otherwise (0)
- `diri_alpha`: Dirichlet α
- `tt_ratio`: ratio of test set samples

A full list of configuration parameters and their descriptions are given in `utils/options.py`.

example:

`python3 -u main_dapfl.py --dataset cifar10 --model cnn --num_classes 10 --epochs 200 --lr 0.01  --num_users 100 --gpu 0 --local_ep 1 --frac 0.2 --local_bs 10 --gen_data 0 --diri_alpha 0.5  --tt_ratio 0.1 `


# Acknowledgements

Much of the code in this repository was adapted from code in the repository https://github.com/lgcollins/FedRep/tree/main by Liam Collins et al. , which in turn was adapted from https://github.com/pliang279/LG-FedAvg by Paul Pu Liang et al. and https://github.com/shaoxiongji/federated-learning by Shaoxiong Ji. 

