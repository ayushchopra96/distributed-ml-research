# Federated Continual Learning

This is an official implementation of Federated Continual Learning with Adaptive Parameter Communication ([paper](https://arxiv.org/abs/2003.03196)). We propose a novel federated continual learning framework, Federated continual learning with Adaptive Parameter Communication (Fed-APC), which additively decomposes the network weights into global shared parameters and sparse task-specific parameters to minimize interference between incompatible tasks, and also allows inter-client knowledge transfer by communicating the sparse task-specific parameters.

## Environmental Requirements
```bash
$ pip install -r requirements.txt
```

## Generating Tasks
```bash
$ cd data
$ python3 data_generator.py
```
At first execution, you will download `8 heterogeneous dataset`, including SVHN, TrafficSigns, FaceScrub, MNIST, FashionMNIST, NotMNIST, CIFAR-10, and CIFAR-100, which take few minutes. 
> ***Note***: if you (accidently) stop downloading during processing, I recommend to remove all files in `data/mixture_loader/` before you download again due to some malfunctioning of third party module.

## Training Models
```bash
$ cd [root/of/the/project]
$ python3 main.py -g [gpu_id,] -gc [num_clients_per_gpu_id,]
```
Example) Run 3 models on gpu 0,1,2 repectively (9 models in total)
```bash
$ python3 main.py -g 0,1,2 -gc 3,3,3
```
> ***Note***: Since we upload a task pool, `non-iid-50`, that supports at most 5 clients and 10 tasks per client, please run no more than `5 clients` in total.

## Checking Performance
```bash
$  cd outputs/logs/ # open corresponding logs according to date and time 
```
