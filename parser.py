##################################
# Parser for launching application
##################################
__author__ = "Wonyong Jeong"
__credits__ = ["Wonyong Jeong"]
__email__ = "wyjeong@kaist.ac.kr"
##################################

import argparse

MODEL = 'fed-apc'
BASE_NETWORK = 'alexnet-like'
FRAC_CLIENTS = 1.0

NUM_ROUNDS = 20
NUM_EPCOHS = 1
BATCH_SIZE = 100

SPARSE_COMMUNICATION = False
CLIENT_SPARSITY = 0.3
SERVER_SPARSITY = 0.3 

TASK_POOL = 'non_iid_50'
TASK_POOL_PATH = 'data/tasks/'
MANUAL_TASKS = []

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
    
    def set_arguments(self):
        self.parser.add_argument('-g', '--gpu', required=True, type=str, help='gpu id to load models, i.e, -g 0 if you want to use one gpu with id 0 or -g 1,2,3 if you want to use 3 gpus with id 0,1,2')
        self.parser.add_argument('-gc', '--gpu-clients', required=True, type=str, help='# clients per gpu, i.e., -g 0 -gc 3 if you want to load 3 models on gpu 0')
        self.parser.add_argument('-m', '--exp-mark', type=str, default='', help='optional, this keyword will be append at the end of the output file')
        self.parser.add_argument('--model', type=str, default=MODEL, help='model to experiment (fed-apc)')
        self.parser.add_argument('--frac-clients', type=float, default=FRAC_CLIENTS, help='fraction of connected clients per round for parameter aggregation')
        self.parser.add_argument('--base-network', type=str, default=BASE_NETWORK, help='alexnet-like, etc.')
        self.parser.add_argument('--num-rounds', type=int, default=NUM_ROUNDS, help='number of rounds per task')
        self.parser.add_argument('--num-epochs', type=int, default=NUM_EPCOHS, help='number of epochs per round')
        self.parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='minibatch size')
        self.parser.add_argument('--sparse-comm', type=bool, default=SPARSE_COMMUNICATION, help='option for sparsification of parameters per communicationn')
        self.parser.add_argument('--client-sparsity', type=float, default=CLIENT_SPARSITY,  help='sparsity of client-server transmission')
        self.parser.add_argument('--server-sparsity', type=float, default=SERVER_SPARSITY,  help='sparsity of server-client transmission')
        self.parser.add_argument('--task-pool', type=str, default=TASK_POOL, help='non_iid_50, etc.')
        self.parser.add_argument('--task-path', type=str, default=TASK_POOL_PATH, help='path to save task files')
        self.parser.add_argument('--manual-tasks', type=str, default=MANUAL_TASKS, help='manual tasks', nargs='*')
    
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
