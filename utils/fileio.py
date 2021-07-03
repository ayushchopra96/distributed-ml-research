##################################
# utils/data
# : common functions for file io
##################################
__author__ = "Wonyong Jeong"
__credits__ = ["Wonyong Jeong"]
__email__ = "wyjeong@kaist.ac.kr"
##################################

import os
import json
import numpy as np
from datetime import datetime

def np_save(base_dir, filename, data):
    if os.path.isdir(base_dir) == False:
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)

def save_task(base_dir, filename, data):
    np_save(base_dir, filename, data)

def save_weights(base_dir, filename, weights):
    np_save(base_dir, filename, weights)

def write_file(filepath, filename, data):
    if os.path.isdir(filepath) == False:
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile)

def np_load(path):
    return np.load(path, allow_pickle=True)

def load_task(base_dir, task):
    return np_load(os.path.join(base_dir, task))

def load_weights(path):
    return np_load(path)
    
def console_log(message):
    print('[%s] %s' %(datetime.now().strftime("%Y/%m/%d-%H:%M:%S"), message))

def write_file(filepath, filename, data):
    if os.path.isdir(filepath) == False:
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile)
