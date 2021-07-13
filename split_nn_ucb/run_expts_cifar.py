import multiprocessing as mp
import os
commands = [
    # UCB-CS + Polling
    '''
    CUDA_VISIBLE_DEVICES=0 python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 2 --discount 0.7 \
    --poll_clients \
    --batch_size 64 \
    --epochs 200
    ''',

    # UCB-CS + No Polling
    '''
    CUDA_VISIBLE_DEVICES=1 python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 2 --discount 0.7 \
    --batch_size 64 \
    --epochs 200
    ''',

    # UCB-CS + Polling + Interrupted
    '''
    CUDA_VISIBLE_DEVICES=2 python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 2 --discount 0.7 \
    --poll_clients \
    --interrupted \
    --batch_size 64 \
    --epochs 200
    ''',

    # UCB-CS + No Polling + Interrupted
    '''
    CUDA_VISIBLE_DEVICES=3 python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 2 --discount 0.7 \
    --interrupted \
    --batch_size 64 \
    --epochs 200
    '''
]


def run_command(command):
    os.system(command)


with mp.Pool(4) as p:
    p.map(run_command, commands)
