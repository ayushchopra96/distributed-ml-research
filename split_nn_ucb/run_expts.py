import os
import multiprocessing as mp

NUM_GPUS = 1
JOBS_PER_GPU = 2

cesl_args = " --interrupted --use_ucb --use_masked --use_additive --use_contrastive --epochs 20 --use_lenet --use_head --lr 1e-3 "
cesl_random_args = " --interrupted --use_random --use_masked --use_additive --use_contrastive --epochs 20 --use_head --use_lenet --lr 1e-3 "
vanilla_args = " --vanilla "

def run_command(args):
    i, command = args
    gpu_id = i % NUM_GPUS
    assert(command.find("local_parallel") != -1)
    os.system(f"CUDA_VISIBLE_DEVICES={gpu_id} {command}")

# Client Model Size Ablation
def make_command_client_size(args):
    job_id, client_size = args
    command = f'''CUDA_VISIBLE_DEVICES={job_id % NUM_GPUS} python3 train.py --non_iid_50_v1 --num_clients 5 --k 3 --experiment_name client_size_expt_{client_size}_{job_id} --client_model_size {client_size} --local_parallel '''
    command += cesl_args
    os.system(command)

client_sizes = [1, 2, 3, 4] * 5

# with mp.Pool(NUM_GPUS * JOBS_PER_GPU) as p:
#     p.map(make_command_client_size, enumerate(client_sizes))

# Interrupt Range Ablation
def make_command_interrupt_range(args):
    job_id, interrupt_range = args
    command = f'''CUDA_VISIBLE_DEVICES={job_id % NUM_GPUS} python3 train.py --non_iid_50_v1 --num_clients 5 --k 3 --experiment_name interrupt_range_expt_{interrupt_range}_{job_id} --interrupt_range {interrupt_range} --local_parallel '''
    command += cesl_args
    print(command)
    os.system(command)
    

interrupt_ranges = [0.3, 0.45, 0.6, 0.75, 0.9] * 5

# with mp.Pool(NUM_GPUS * JOBS_PER_GPU) as p:
#     p.map(make_command_interrupt_range, enumerate(interrupt_ranges))

# sparsity_weights = [0., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] * 5
sparsity_weights = [5e-6, 1e-6, 1e-7] * 5

# Feature Sparsity Ablation
def make_command_sparsity(args):
    job_id, sparsity = args
    command = f'''CUDA_VISIBLE_DEVICES={job_id % NUM_GPUS} python3 train.py --non_iid_50_v1 --num_clients 5 --k 3 --experiment_name feature_sparsity_expt_{sparsity}_{job_id} --sparse_features --sparsity_weight {sparsity} --lr 1e-3 '''
    command += cesl_args
    os.system(command)

# with mp.Pool(NUM_GPUS * JOBS_PER_GPU) as p:
#     p.map(make_command_sparsity, enumerate(sparsity_weights))

# non_iid_50 expts
def make_command_niid_50(random_or_not, vanilla_or_not, num_clients, k, avg=False, masked=False, run=0, use_head=True, version="v1", local_parallel=False, interrupt=0.6):
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
    if local_parallel:
        name = f"LocalParallel-{interrupt}-{k}-{num_clients}-{random_or_not}"
        extra += f" --local_parallel --interrupt_range {interrupt}"
    if use_head == False:
        extra = extra.replace("--use_head", "")
        name = f"CESL-No-Head-{num_clients}-{num_clients}"
    if masked: 
        command = f'''python3 train.py --non_iid_50_{version} --num_clients {num_clients} --k {num_clients} --experiment_name non_iid_50_{version}_Only-Masked_{run} --l1_norm_weight 1e-3 --epochs 20 --use_masked --local_parallel --use_additive --use_lenet --lr 1e-3 ''' 
    else:        
        command = f'''python3 train.py  --non_iid_50_{version} --num_clients {num_clients} --k {k} --experiment_name non_iid_50_{version}_{name}_{run} --l1_norm_weight 1e-3 --use_lenet ''' 
        command += extra
    return command

niid_commands = [
    # Random
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=2, run=0, local_parallel=True),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=2, run=1, local_parallel=True),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=2, run=2, local_parallel=True),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=2, run=3, local_parallel=True),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=5, k=2, run=4, local_parallel=True),
    # Bandit
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=2, run=0, local_parallel=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=2, run=1, local_parallel=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=2, run=2, local_parallel=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=2, run=3, local_parallel=True),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=2, run=4, local_parallel=True),
    # # Vanilla
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=0),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=1),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=2),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=3),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=4),
    # # SplitFed
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=0, avg=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=1, avg=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=2, avg=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=3, avg=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=4, avg=True),
    # # use_head = False
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=0, use_head=False),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=1, use_head=False),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=2, use_head=False),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=3, use_head=False),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=4, use_head=False),    
    # # Only Masked
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=0, masked=True, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=1, masked=True, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=2, masked=True, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=3, masked=True, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=4, masked=True, local_parallel=True),    
    # # local_parallel = True
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=0, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=1, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=2, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=3, local_parallel=True),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=4, local_parallel=True),    
]

with mp.Pool(2 * NUM_GPUS) as p:
    p.map(run_command, enumerate(niid_commands))

# # Cifar10 expts
def make_command_cifar(random_or_not, vanilla_or_not, num_clients, k, avg=False, masked=False, run=0, use_head=True, version="v1", local_parallel=False, interrupt=0.6):
    extra = ""
    if avg:
        name = f"SplitFed-{num_clients}-{num_clients}"
        extra += " --avg_clients --vanilla "
    elif local_parallel:
        name = f"LocalParallel-{interrupt}-{k}-{num_clients}"
        extra += cesl_args
        extra += f" --local_parallel --interrupt_range {interrupt}"
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
        name += "No-Head"
    if masked: 
        command = f'''python3 train.py --cifar --classwise_subset --num_clients {num_clients} --k {num_clients} --experiment_name cifar10_NIID_{version}_Only-Masked_{run} --epochs 20 --use_masked --l1_norm_weight 1e-4 --lr 1e-3  --alpha 0. --use_lenet --local_parallel ''' 
    else:        
        command = f'''python3 train.py  --cifar --classwise_subset --num_clients {num_clients} --k {k} --experiment_name cifar10_NIID_{version}_{name}_{run}  --l1_norm_weight 1e-4 --lr 1e-3 --alpha 0. --use_lenet --local_parallel ''' 
        command += extra
    # command = command.replace("--use_lenet", "")
    command = command.replace("--use_additive", "")
    return command

niid_commands = [
    # Random
    make_command_cifar(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=0, local_parallel=True),
    make_command_cifar(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=1, local_parallel=True),
    make_command_cifar(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=2, local_parallel=True),
    make_command_cifar(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=3, local_parallel=True),
    make_command_cifar(random_or_not=True, vanilla_or_not=False, num_clients=5, k=3, run=4, local_parallel=True),
    # # Bandit
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=0, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=1, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=2, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=3, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=4, local_parallel=True),
    # # Vanilla
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=0),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=1),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=2),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=3),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=4),
    # # SplitFed
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=0, avg=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=1, avg=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=2, avg=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=3, avg=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=True, num_clients=5, k=5, run=4, avg=True),
    # # use_head = False
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=0, use_head=False),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=1, use_head=False),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=2, use_head=False),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=3, use_head=False),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=3, run=4, use_head=False),
    # # Only Masked
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=0, masked=True, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=1, masked=True, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=2, masked=True, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=3, masked=True, local_parallel=True),
    # make_command_cifar(random_or_not=False, vanilla_or_not=False, num_clients=5, k=5, run=4, masked=True, local_parallel=True),
]

# with mp.Pool(4) as p:
#     p.map(run_command, enumerate(niid_commands))
