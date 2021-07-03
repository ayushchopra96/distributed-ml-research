##################################
# Main Process
# : initiate Server application that  
# initialize multiple clients threads
# and model hyper-parameters
##################################
__author__ = "Wonyong Jeong"
__credits__ = ["Wonyong Jeong"]
__email__ = "wyjeong@kaist.ac.kr"
##################################

import os
from datetime import datetime
from parser import Parser
from utils.data import *

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu
    opt = set_options(opt)
    if opt.model == 'fed-apc': 
        from models.fed_apc.server import Server
        server = Server(opt)
        server.run()
    else:
        print('incorrect model was given: {}'.format(opt.model))
        os._exit(0)

def set_options(opt):
    now = datetime.now().strftime("%Y%m%d-%H%M")
    opt.log_dir = 'outputs/logs/{}'.format(now)
    if len(opt.exp_mark)>0:
        opt.log_dir += '-{}'.format(opt.exp_mark)        
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    # adaptive learning rate
    opt.lr_patience = 3
    opt.lr_factor = 3
    opt.lr_min = 1e-8
    # base network hyperparams
    if opt.base_network == 'alexnet-like':
        opt.lr = 1e-4
        opt.wd = 1e-2
        opt.momentum = 0.9
    if 'fed-apc' in opt.model:
        opt.wd = 1e-4 
        opt.lambda_l1 = 1e-3
        opt.lambda_l2 = 100.
        opt.lambda_mask = 0
    # task pool options
    opt = set_task_pool(opt)
    return opt
if __name__ == '__main__':
    main(Parser().parse())

