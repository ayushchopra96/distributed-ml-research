import os
import multiprocessing as mp

NUM_GPUS = 1
JOBS_PER_GPU = 5

cesl_args = " --interrupted --use_vw --use_masked --use_contrastive --poll_clients "
cesl_random_args = " --interrupted --use_random --use_masked --use_contrastive --poll_clients "
vanilla_args = " --alpha 0.0 --vanilla "

def run_command(command):
    os.system(command)

# Interrupt Range Ablation
def make_command_interrupt_range(args):
    job_id, interrupt_range = args
    command = f'''CUDA_VISIBLE_DEVICES={job_id % NUM_GPUS} python3 train.py --cifar --num_clients 10 --k 6 --experiment_name interrupt_range_expt_{interrupt_range} --interrupt_range {interrupt_range} --classwise_subset'''
    command += "--interrupted --use_masked --use_contrastive --poll_clients "
    os.system(command)
    

interrupt_ranges = [0.3, 0.45, 0.6, 0.75]

# with mp.Pool(NUM_GPUS * JOBS_PER_GPU) as p:
#     p.map(make_command_interrupt_range, enumerate(interrupt_ranges))

# non_iid_50 expts
def make_command_niid_50(random_or_not, vanilla_or_not, num_clients, k):
    if vanilla_or_not:
        name = f"Vanilla-{num_clients}/{num_clients}"
    else:
        if random_or_not:
            name = f"CESL-Random-{k}/{num_clients}"
        else:
            name = f"CESL-{k}/{num_clients}"
    command = f'''python3 train.py --cifar --non_iid_50 --num_clients {num_clients} --k {k} --experiment_name non_iid_50_{name} ''' 
    return command

niid_commands = [
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=6),
    make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=50, k=30),
    # make_command_niid_50(random_or_not=True, vanilla_or_not=False, num_clients=50, k=30),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=6),
    make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=50, k=30),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=False, num_clients=50, k=15),
    make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=10, k=10),
    # make_command_niid_50(random_or_not=False, vanilla_or_not=True, num_clients=50, k=50),
]

# with mp.Pool(5) as p:
#     p.map(run_command, niid_commands)


# # Cifar10 expts
def make_command_cifar(random_or_not, vanilla_or_not, num_clients, k, iid_or_not):
    extra_args = ""
    #if num_clients == 100:
    #    extra_args += " --num_groups "
    if vanilla_or_not:
        name = f"Vanilla-{num_clients}-{num_clients}"
        extra_args += " --vanilla "
    else:
        if random_or_not:
            name = f"CESL-Random-{k}-{num_clients}"
            extra_args += cesl_random_args
        else:
            name = f"CESL-{k}-{num_clients}"
            extra_args += cesl_args
    if iid_or_not:
        name = "IID-" + name
    else:
        extra_args += " --classwise_subset"
        name = "NIID-" + name
    command = f'''python3 train.py --cifar --num_clients {num_clients} --k {k} --experiment_name cifar_{name} ''' 
    command += extra_args
    return command

niid_cifar_commands = [
    # Non-IID Cifar
    make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=10, k=6),
    make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=100, k=30),
    # make_command_cifar(iid_or_not=False, random_or_not=True, vanilla_or_not=False, num_clients=100, k=60),
    make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=10, k=6),
    make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=10, k=3),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=100, k=30),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=False, num_clients=100, k=60),
    make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=True, num_clients=10, k=10),
    # make_command_cifar(iid_or_not=False, random_or_not=False, vanilla_or_not=True, num_clients=100, k=100),
]

with mp.Pool(5) as p:
    p.map(run_command, niid_cifar_commands)

iid_cifar_commands = [
    # IID Cifar
    make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=10, k=6),
    make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=10, k=3),
    make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=100, k=30),
    make_command_cifar(iid_or_not=True, random_or_not=True, vanilla_or_not=False, num_clients=100, k=60),
    make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=10, k=6),
    make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=10, k=3),
    make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=100, k=30),
    make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=False, num_clients=100, k=60),
    make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=True, num_clients=10, k=6),
    make_command_cifar(iid_or_not=True, random_or_not=False, vanilla_or_not=True, num_clients=100, k=30),
]

#with mp.Pool(5) as p:
#    p.map(run_command, iid_cifar_commands)