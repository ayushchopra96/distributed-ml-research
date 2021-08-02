import multiprocessing as mp
import os

NUM_GPUS = 8

commands = [
    # P = 5
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --batch_size 32 \
    --epochs 150 \
    --num_partitions 5
    ''',

    # P = 3
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --batch_size 32 \
    --epochs 150 \
    --num_partitions 3
    ''',

    # P = 1
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --batch_size 32 \
    --epochs 150 \
    --num_partitions 1
    ''',

    # Classwise Subset + P = 1
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --batch_size 32 \
    --epochs 150 \
    --classwise_subset \
    --num_partitions 3
    ''',

    # Classwise Subset + P = 5
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --batch_size 32 \
    --epochs 150 \
    --classwise_subset \
    --num_partitions 5
    ''',

    # Classwise Subset + P = 3
    '''
    CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py \
    --cifar \
    --num_clients 10 \
    --batch_size 32 \
    --epochs 150 \
    --classwise_subset \
    --num_partitions 3
    ''',
]


def run_command(args):
    command, id = args
    id = id % NUM_GPUS
    os.system(command.format(id))


with mp.Pool(2 * NUM_GPUS) as p:
    p.map(run_command, zip(commands, range(len(commands))))
