##################################
# Data/Task Loader
# : loads tasks according to the given 
# squence per client for every rounds that 
# are specified in "num_rounds" 
##################################
__author__ = "Wonyong Jeong"
__credits__ = ["Wonyong Jeong"]
__email__ = "wyjeong@kaist.ac.kr"
##################################

import os
import pdb
import glob
import numpy as np
from utils.fileio import *
from utils.data import *

class DataLoader:
    def __init__(self, opt, client_id=None):
        self.opt = opt
        self.tasks=[]
        self.client_id = client_id
        self.base_dir = os.path.join(self.opt.task_path, self.opt.task_pool) 
        if not client_id == None:
            self.load_tasks()

    def load_tasks(self):
        if len(self.opt.manual_tasks)>0:
            self.tasks = self.opt.manual_tasks
        elif self.opt.task_pool in ['non_iid_50']:
            task_set = {
                0: ['cifar100_5.npy', 'cifar100_13.npy', 'face_scrub_0.npy', 'cifar100_14.npy', 'svhn_1.npy', 'traffic_sign_0.npy', 'not_mnist_1.npy', 'cifar100_8.npy', 'face_scrub_13.npy', 'cifar100_4.npy'], 
                1: ['cifar100_2.npy', 'traffic_sign_5.npy', 'face_scrub_14.npy', 'traffic_sign_4.npy', 'not_mnist_0.npy', 'mnist_0.npy', 'face_scrub_2.npy', 'face_scrub_15.npy', 'cifar100_1.npy', 'fashion_mnist_1.npy'],
                2: ['face_scrub_11.npy', 'svhn_0.npy', 'face_scrub_10.npy', 'face_scrub_6.npy', 'face_scrub_7.npy', 'cifar100_3.npy', 'cifar100_10.npy', 'mnist_1.npy', 'face_scrub_1.npy', 'traffic_sign_1.npy'], 
                3: ['fashion_mnist_0.npy', 'cifar100_15.npy', 'face_scrub_3.npy', 'cifar10_1.npy', 'cifar100_7.npy', 'face_scrub_8.npy', 'cifar10_0.npy', 'face_scrub_9.npy', 'cifar100_0.npy', 'cifar100_6.npy'],
                4: ['traffic_sign_7.npy', 'face_scrub_5.npy', 'traffic_sign_6.npy', 'traffic_sign_3.npy', 'traffic_sign_2.npy', 'cifar100_12.npy', 'cifar100_11.npy', 'cifar100_9.npy', 'face_scrub_12.npy', 'face_scrub_4.npy'] 
            }
            self.tasks = task_set[self.client_id]
        else:
            print('no correct task_pool was given: {}'.format(self.opt.task_pool))
            os._exit(0)

    def get_info(self):
        return {'num_tasks': len(self.tasks)}
    
    def get_task(self, task_id):
        if len(self.opt.manual_tasks)>0:
            task = load_task('', self.tasks[task_id])
        else:
            task = load_task(self.base_dir, self.tasks[task_id])
        return task.item()