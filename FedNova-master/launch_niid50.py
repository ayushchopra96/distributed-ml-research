import os
import multiprocessing as mp
num_clients = 10

fednova = """python3 train_non_iid_50.py --pattern constant \
    --lr 0.02 --bs 32 --localE 2 --alpha 0.1 --mu 0 --momentum 0.9 \
    --save -p --name FedNova_momen_baseline --optimizer fednova --model VGG \
    --rank {} --size 16 --backend mpi \
    --rounds 100 --seed 3 --NIID --print_freq 50"""


def run_fednova(rank):
    os.system(fednova.format(rank))


with mp.Pool(num_clients) as p:
    p.map(run_fednova, list(range(num_clients)))
