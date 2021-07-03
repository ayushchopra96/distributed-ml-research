##################################
# utils/data
# : common functions for data processing
##################################
__author__ = "Wonyong Jeong"
__credits__ = ["Wonyong Jeong"]
__email__ = "wyjeong@kaist.ac.kr"
##################################

import os
import random
from datetime import datetime 

def random_shuffle(seed, _list):
    random.seed(seed)
    random.shuffle(_list)

def random_sample(seed, _list, num_pick):
    random.seed(seed)
    return random.sample(_list, num_pick)

def random_int(seed, start, end):
    random.seed(seed)
    random.randint(start, end)

def set_task_pool(opt):
    if opt.task_pool == 'non_iid_50':
        opt.datasets    = [0, 1, 2, 3, 4, 5, 6, 7] 
        opt.num_clients = 5
        opt.num_tasks   = 10
        opt.num_classes = 5
    else:
        print('no correct task_pool was given: {}'.format(opt.task_pool))
        os._exit(0)
    return  opt

def get_dataset_name(dataset_id):
    if dataset_id == 0:
        return 'cifar10'
    elif dataset_id == 1:
        return 'cifar100'
    elif dataset_id == 2:
        return 'mnist'
    elif dataset_id == 3:
        return 'svhn'
    elif dataset_id == 4:
        return 'fashion_mnist'
    elif dataset_id == 5:
        return 'traffic_sign'
    elif dataset_id == 6:
        return 'face_scrub'
    elif dataset_id == 7:
        return 'not_mnist'
