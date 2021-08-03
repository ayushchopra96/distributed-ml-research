#

import multiprocessing as mp
import os

NUM_GPUS = 4

commands = [
    # Random 3 - IID
    "python3 loop_flipped_10.py --cifar --use_random --k 3 --experiment_name random_iid_3",

    # Random 5 - IID
    "python3 loop_flipped_10.py --cifar --use_random --k 5 --experiment_name random_iid_5",

    # Random 7 - IID
    "python3 loop_flipped_10.py --cifar --use_random --k 7  --experiment_name random_iid_7",

    # BayesianUCB 3 - IID
    "python3 loop_flipped_10.py --cifar --use_ucb --k 3 --experiment_name ucb_iid_3",

    # BayesianUCB 5 - IID
    "python3 loop_flipped_10.py --cifar --use_ucb --k 5 --experiment_name ucb_iid_5",

    # BayesianUCB 7 - IID
    "python3 loop_flipped_10.py --cifar --use_ucb --k 7  --experiment_name ucb_iid_7",

    # Random 3 - NIID
    "python3 loop_flipped_10.py --cifar --classwise_subset --use_masked --interrupted --use_contrastive --use_random --k 3 --experiment_name random_niid_3",

    # Random 5 - NIID
    "python3 loop_flipped_10.py --cifar --use_random --k 5 --use_masked --interrupted --use_contrastive --experiment_name random_niid_5",

    # Random 7 - NIID
    "python3 loop_flipped_10.py --cifar --use_random --k 7 --use_masked --interrupted --use_contrastive  --experiment_name random_niid_7",

    # BayesianUCB 3 - NIID
    "python3 loop_flipped_10.py --cifar --use_ucb --k 3 --use_masked --interrupted --use_contrastive --experiment_name ucb_niid_3",

    # BayesianUCB 5 - NIID
    "python3 loop_flipped_10.py --cifar --use_ucb --k 5 --use_masked --interrupted --use_contrastive --experiment_name ucb_niid_5",

    # BayesianUCB 7 - NIID
    "python3 loop_flipped_10.py --cifar --use_ucb --k 7 --use_masked --interrupted --use_contrastive --experiment_name ucb_niid_7",
]


def run_command(args):
    command, id = args
    id = id % NUM_GPUS
    os.system(command.format(id))


with mp.Pool(2 * NUM_GPUS) as p:
    p.map(run_command, zip(commands, range(len(commands))))
