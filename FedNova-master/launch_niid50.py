import os
import multiprocessing as mp
import random

num_clients = 1


fednova = """python3 -m torch.distributed.launch --master_port={} launch_exp.py {}"""


def run_fednova(rank):
    os.system(fednova.format(random.randint(0, 10000), rank))


with mp.Pool(num_clients) as p:
    p.map(run_fednova, list(range(num_clients)))
