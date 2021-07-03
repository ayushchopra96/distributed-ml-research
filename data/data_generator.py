##################################
# Data/Task Generator 
# : generates squence of tasks namely, 
# non-iid-50, which consists of 50 tasks from 
# 8 heterogenous datasets including SVHN, MNIST, 
# Fashion-MNIST, Not-MNIST, FaceScrub, TrafficSigns, Cifar-10 & 100
##################################
__author__ = "Wonyong Jeong"
__credits__ = ["Wonyong Jeong"]
__email__ = "wyjeong@kaist.ac.kr"
##################################

import os
import pdb
import cv2
import sys
import argparse
import random
import numpy as np
import tensorflow as tf

sys.path.insert(0,'..')
from utils.data import *
from utils.fileio import *
from third_party.mixture_loader.mixture import *

TASK_POOL = 'non_iid_50'
TASK_PATH = '../data/tasks/'

class DataParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
    def set_arguments(self):
        self.parser.add_argument('--task-pool', type=str, default=TASK_POOL, help='non_iid_50, etc.')
        self.parser.add_argument('--task-path', type=str, default=TASK_PATH, help='path to save task files')
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args

class DataGenerator:
    def __init__(self, opt):
        self.opt = set_task_pool(opt)
        self.fixed_global_random_seed = 1
        self.seprate_ratio = (0.7, 0.2, 0.1) # train, test, valid
        self.mixture_dir = '../data/mixture_loader/saved'
        self.mixture_filename = 'saved_mixture.npy'
        self.base_dir = os.path.join(self.opt.task_path, self.opt.task_pool) 
        self.generate_data()

    def generate_data(self):
        saved_mixture_filepath = os.path.join(self.mixture_dir, self.mixture_filename)
        if os.path.exists(saved_mixture_filepath):
            print('loading mixture data: {}'.format(saved_mixture_filepath))
            mixture = np.load(saved_mixture_filepath, allow_pickle=True)
        else:
            print('downloading & processing mixture data')
            mixture = get(base_dir=self.mixture_dir, fixed_order=True)
            np_save(self.mixture_dir, self.mixture_filename, mixture)
        self.generate_tasks(mixture)

    def generate_tasks(self, mixture):
        print('generating tasks ...')
        self.task_cnt = -1
        for dataset_id in self.opt.datasets:
            self._generate_tasks(dataset_id, mixture[0][dataset_id])
    
    def _generate_tasks(self, dataset_id, data):
        # concat train & test
        x_train = data['train']['x']
        y_train = data['train']['y']
        x_test = data['test']['x']
        y_test = data['test']['y']
        x_valid = data['valid']['x']
        y_valid = data['valid']['y']
        x = np.concatenate([x_train, x_test, x_valid])
        y = np.concatenate([y_train, y_test, y_valid])
        # shuffle dataset
        idx_shuffled = np.arange(len(x))
        random_shuffle(self.fixed_global_random_seed, idx_shuffled)
        x = x[idx_shuffled]
        y = y[idx_shuffled]
        if self.opt.task_pool == 'non_iid_50':
            self._generate_non_iid_50(dataset_id, x, y)

    def _generate_non_iid_50(self, dataset_id, x, y):
        labels = np.unique(y)
        random_shuffle(self.fixed_global_random_seed, labels)
        labels_per_task = [labels[i:i+self.opt.num_classes] for i in range(0, len(labels), self.opt.num_classes)]
        for task_id, _labels in enumerate(labels_per_task):
            if dataset_id == 5 and task_id == 8:
                continue
            elif dataset_id in [1,6] and task_id > 15:
                continue
            self.task_cnt += 1
            idx = np.concatenate([np.where(y[:]==c)[0] for c in _labels], axis=0)
            random_shuffle(self.fixed_global_random_seed, idx)
            x_task = x[idx]
            y_task = y[idx]

            idx_labels = [np.where(y_task[:]==c)[0] for c in _labels]
            for i, idx_label in enumerate(idx_labels):
                y_task[idx_label] = i # reset class_id 
            y_task = tf.keras.utils.to_categorical(y_task, len(_labels))
            
            filename = '{}_{}'.format(get_dataset_name(dataset_id), task_id)
            self._save_task(x_task, y_task, _labels, filename)

    def _save_task(self, x_task, y_task, labels, filename):
        print('filename: {}, labels: [{}], num_examples: {}'\
                .format(filename,', '.join(map(str, labels)), len(x_task)))
        pairs = list(zip(x_task, y_task))
        num_examples = len(pairs)
        num_train = int(num_examples*self.seprate_ratio[0]) 
        num_test = int(num_examples*self.seprate_ratio[1])  
        save_task(base_dir=self.base_dir, filename=filename, data={
            'train': pairs[:num_train],
            'test' : pairs[num_train:num_train+num_test],
            'valid': pairs[num_train+num_test:],
            'labels': labels,
            'name': filename,
        })

if __name__ == '__main__':
    DataGenerator(DataParser().parse())
