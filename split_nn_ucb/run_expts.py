import os
import multiprocessing as mp

NUM_GPUS = 1
JOBS_PER_GPU = 1

cesl_args = " --interrupted --use_ucb --use_masked --use_additive --use_contrastive --epochs 20 --use_lenet --use_head "
cesl_random_args = " --interrupted --use_random --use_masked --use_additive --use_contrastive --epochs 20 --use_lenet --use_head "
vanilla_args = " --vanilla --use_head "

def run_command(args):
    i, command = args
    gpu_id = i % NUM_GPUS
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} {command}")

# Interrupt Range Ablation
def make_command_interrupt_range(args):
    job_id, interrupt_range = args
    command = f'''CUDA_VISIBLE_DEVICES={job_id % NUM_GPUS} python3 train.py --cifar --num_clients 10 --k 6 --experiment_name interrupt_range_iid_expt_{interrupt_range} --interrupt_range {interrupt_range} '''
    command += " --interrupted --use_masked --use_contrastive --use_ucb --epochs 20"
    os.system(command)
    

interrupt_ranges = [0.3, 0.45, 0.6, 0.75, 0.9]

# with mp.Pool(NUM_GPUS * JOBS_PER_GPU) as p:
#     p.map(make_command_interrupt_range, enumerate(interrupt_ranges))

# non_iid_50 expts
def make_command_niid_50(random_or_not, vanilla_or_not, num_clients, k, avg=False, masked=False, run=0, use_head=True, version="v1"):
    extra = ""
    if avg:
        name = f"SplitFed-{num_clients}-{num_clients}"
        extra += " --avg_clients --vanilla "
    else:
        if vanilla_or_not:
            name = f"Vanilla-{num_clients}-{num_clients}"
            extra += vanilla_args
        else:
            if random_or_not:
                name = f"CESL-Random-{k}-{num_clients}"
                extra += cesl_random_args
            else:
                name = f"CESL-{k}-{num_clients}"
                extra += cesl_args
    if use_head == False:
        extra = extra.replace("--use_head", "")
        name = f"CESL-No-Head-{num_clients}-{num_clients}"
    if masked: 
        command = f'''python3 train.py --non_iid_50_{version} --num_clients {num_clients} --k {num_clients} --experiment_name non_iid_50_{version}_Only-Masked_{run} --l1_norm_weight 1e-3 --epochs 20 --use_masked --use_additive --use_lenet ''' 
    else:        
        command = f'''python3 train.py  --non_iid_50_{version} --num_clients {num_clients} --k {k} --experiment_name non_iid_50_{version}_{name}_{run} --l1_norm_weight 1e-3 --use_lenet ''' 
        command += extra
    return command

niid_commands = [
    # Random
    # make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6, run=0),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6, run=1),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6, run=2),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6, run=3),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6, run=4),
    # No Client Selection
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=10),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5),
    # Bandit
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=3),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6, run=0),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6, run=1),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6, run=2),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6, run=3),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6, run=4),
    # Vanilla
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=0),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=1),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=2),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=3),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=4),
    # SplitFed
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=0, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=1, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=2, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=3, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10, run=4, avg=True),
    # # use_head = False
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=6, run=0, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=6, run=1, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=6, run=2, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=6, run=3, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=6, run=4, use_head=False),
    
    # # Only Masked
    # # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=10, masked=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=0),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=1),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=2),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=3),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=4),
]

# with mp.Pool(1 * NUM_GPUS) as p:
#     p.map(run_command, enumerate(niid_commands))

niid_commands = [
    # Random
    # make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=0),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=1),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=2),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=3),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=4),
    # No Client Selection
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=10),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5),
    # Bandit
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=3),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=0),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=1),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=2),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=3),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=4),
    # Vanilla
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=0),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=1),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=2),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=3),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=4),
    # SplitFed
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=0, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=1, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=2, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=3, avg=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=4, avg=True),
    # # use_head = False
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=3, run=0, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=3, run=1, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=3, run=2, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=3, run=3, use_head=False),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=3, run=4, use_head=False),
    
    # # Only Masked
    # # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=10, masked=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=0),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=1),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=2),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=3),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, masked=True, run=4),
]

with mp.Pool(2 * NUM_GPUS) as p:
    p.map(run_command, enumerate(niid_commands))

# # Cifar10 expts
def make_command_cifar(random_or_not, vanilla_or_not, num_clients, k, iid_or_not, avg_clients=False):
    extra_args = ""
    #if num_clients == 100:
    #    extra_args += " --num_groups "
    if vanilla_or_not is None:
        name = f"CESL-no-bandit-{num_clients}-{num_clients}"
        extra_args += " --interrupted --use_masked --use_additive --use_contrastive --interrupt_range 0.3 --epochs 20 "
    elif vanilla_or_not == True:
        name = f"Vanilla-{num_clients}-{num_clients}"
        extra_args += " --vanilla "
    elif vanilla_or_not == False:
        if random_or_not:
            name = f"CESL-Random-{k}-{num_clients}"
            extra_args += cesl_random_args
        else:
            name = f"CESL-{k}-{num_clients}"
            extra_args += cesl_args
    if avg_clients:
        extra_args += " --avg_clients "
        name = "FedSplit-IID-" + name
    if not iid_or_not:
        extra_args += " --classwise_subset"
        name = "NIID-" + name
    else:
        name = "IID-" + name
    command = f'''python3 train.py --cifar --num_clients {num_clients} --k {k} --experiment_name cifar_{name} ''' 
    command += extra_args
    return command

niid_cifar_commands = [
    # Non-IID Cifar
    # make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=10, k=6),
    # make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=100, k=30),
    # make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=100, k=60),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=10, k=6),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=100, k=30),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=100, k=60),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=True, num_clients=10, k=10),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=True, num_clients=100, k=100),
    make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=True, num_clients=10, k=10),
]

# with mp.Pool(3) as p:
#     p.map(run_command, enumerate(niid_cifar_commands))

iid_cifar_commands = [
    # IID Cifar
    # make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=10, k=6),
    # make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=100, k=30),
    # make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=100, k=60),
    # make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=10, k=6),
    # make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=100, k=30),
    # make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=100, k=60),
    # make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=True, num_clients=10, k=10),
    make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=None, num_clients=10, k=10),
    # make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=True, num_clients=100, k=30),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=10),
]

# with mp.Pool(3) as p:
#     p.map(run_command, enumerate(iid_cifar_commands))