import multiprocessing as mp
import os

NUM_GPUS = 8

commands = [
    # UCB-CS + Polling
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --poll_clients \
    --use_ucb \
    --batch_size 32 \
    --epochs 150
    ''',

    # UCB-CS + No Polling
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --use_ucb \
    --batch_size 32 \
    --epochs 150
    ''',

    # UCB-CS + Polling + Interrupted
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --poll_clients \
    --use_ucb \
    --interrupted \
    --batch_size 32 \
    --epochs 150
    ''',

    # UCB-CS + No Polling + Interrupted
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --interrupted \
    --use_ucb \
    --batch_size 32 \
    --epochs 150
    ''',

    # UCB-CS + Polling + constrastive
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --use_contrastive \
    --poll_clients \
    --use_ucb \
    --batch_size 32 \
    --epochs 150
    ''',

    # UCB-CS + No Polling + constrastive
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --use_contrastive \
    --use_ucb \
    --batch_size 32 \
    --epochs 150
    ''',

    # UCB-CS + Polling + Interrupted + constrastive
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --use_contrastive \
    --poll_clients \
    --use_ucb \
    --interrupted \
    --batch_size 32 \
    --epochs 150
    ''',

    # UCB-CS + No Polling + Interrupted + constrastive
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --interrupted \
    --use_ucb \
    --batch_size 32 \
    --epochs 150
    ''',

    # Interrupted only + no contrastive learning
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --interrupted \
    --batch_size 32 \
    --epochs 150
    ''',

    # Interrupted only + contrastive learning
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --k 6 --discount 0.7 \
    --interrupted \
    --use_contrastive \
    --batch_size 32 \
    --epochs 150
    ''',
]


def run_command(args):
    command, id = args
    id = id % NUM_GPUS
    os.system(command.format(id))


with mp.Pool(1) as p:
    p.map(run_command, zip(commands, range(len(commands))))
