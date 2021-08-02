import multiprocessing as mp
import os

NUM_GPUS = 8

def create_command(*args, **kwargs):
    command = "CUDA_VISIBLE_DEVICES={} python3 loop_flipped_10.py "
    for k, v in kwargs.items():
        command += f"--{k} {v} "
    for k in args:
        command += f"--{k} "
    return command

def run_command(args):
    command, id = args
    id = id % NUM_GPUS
    os.system(command.format(id))


commands = [
    ## NUM_CLIENTS = 100
    # Vanilla Split Learning
    create_command("cifar", num_clients=100, batch_size=32, epochs=150, experiment_name="vanilla"),

    # Vanilla Split Learning + Masked + classwise_subset
    create_command("cifar", "use_masked", "classwise_subset", num_groups=5, num_clients=100, batch_size=32, epochs=150, experiment_name="vanilla_classwise_masked"),

    # Masked + classwise_subset + UCB + Polling
    create_command("cifar", "use_ucb", "poll_clients", "use_masked", "classwise_subset", k=6, discount=0.7, num_groups=5, num_clients=100, batch_size=32, epochs=150, experiment_name="classwise_masked_ucb_poll"),

    # Masked + classwise_subset + UCB + Polling + Interrupted + Constrastive
    create_command("cifar", "use_ucb", "poll_clients", "use_masked", "classwise_subset", "interrupted", "contrastive", k=6, discount=0.7, num_groups=5, num_clients=100, batch_size=32, epochs=150, experiment_name="classwise_masked_ucb_poll_interrupted_cl"),

    # Vanilla Split Learning + classwise_subset
    create_command("cifar", "classwise_subset", num_groups=5, num_clients=100, batch_size=32, epochs=150, experiment_name="vanilla_classwise"),

    # Random + constrastive
    create_command("cifar", "use_random", "use_contrastive", k=6, num_clients=100, batch_size=32, epochs=150, experiment_name="random_cl"),

    # Random + Interrupted + constrastive
    create_command("cifar", "use_random", "interrupted", "constrastive", k=6, num_clients=100, batch_size=32, epochs=150, experiment_name="random_interrupted_cl"),

    # UCB-CS + Polling + constrastive
    create_command("cifar", "use_ucb", "use_contrastive", "poll_clients", k=6, discount=0.7, num_clients=100, batch_size=32, epochs=150, experiment_name="ucb_poll_cl"),

    # UCB-CS + Polling + Interrupted + constrastive
    create_command("cifar", "use_ucb", "poll_clients", "interrupted", "use_contrastive", k=6, discount=0.7, num_clients=100, batch_size=32, epochs=150, experiment_name="ucb_poll_interrupted_cl"),

    # Interrupted + contrastive learning
    create_command("cifar", "interrupted", "use_contrastive", num_clients=100, batch_size=32, epochs=150, experiment_name="interrupted_cl"),
]


# with mp.Pool(NUM_GPUS) as p:
#     p.map(run_command, zip(commands, range(len(commands))))

commands = [
    ## NUM_CLIENTS = 10
    # Vanilla Split Learning
    create_command("cifar", num_clients=10, batch_size=32, epochs=150, experiment_name="vanilla"),

    # Vanilla Split Learning + Masked + classwise_subset
    create_command("cifar", "use_masked", "classwise_subset", num_groups=5, num_clients=10, batch_size=32, epochs=150, experiment_name="vanilla_classwise_masked"),

    # Masked + classwise_subset + UCB + Polling
    create_command("cifar", "use_ucb", "poll_clients", "use_masked", "classwise_subset", k=6, discount=0.7, num_groups=5, num_clients=10, batch_size=32, epochs=150, experiment_name="classwise_masked_ucb_poll"),

    # Masked + classwise_subset + UCB + Polling + Interrupted + Constrastive
    create_command("cifar", "use_ucb", "poll_clients", "use_masked", "classwise_subset", "interrupted", "contrastive", k=6, discount=0.7, num_groups=5, num_clients=10, batch_size=32, epochs=150, experiment_name="classwise_masked_ucb_poll_interrupted_cl"),

    # Vanilla Split Learning + classwise_subset
    create_command("cifar", "classwise_subset", num_groups=5, num_clients=10, batch_size=32, epochs=150, experiment_name="vanilla_classwise"),

    # Random + constrastive
    create_command("cifar", "use_random", "use_contrastive", k=6, num_clients=10, batch_size=32, epochs=150, experiment_name="random_cl"),

    # Random + Interrupted + constrastive
    create_command("cifar", "use_random", "interrupted", "constrastive", k=6, num_clients=10, batch_size=32, epochs=150, experiment_name="random_interrupted_cl"),

    # UCB-CS + Polling + constrastive
    create_command("cifar", "use_ucb", "use_contrastive", "poll_clients", k=6, discount=0.7, num_clients=10, batch_size=32, epochs=150, experiment_name="ucb_poll_cl"),

    # UCB-CS + Polling + Interrupted + constrastive
    create_command("cifar", "use_ucb", "poll_clients", "interrupted", "use_contrastive", k=6, discount=0.7, num_clients=10, batch_size=32, epochs=150, experiment_name="ucb_poll_interrupted_cl"),

    # Interrupted + contrastive learning
    create_command("cifar", "interrupted", "use_contrastive", num_clients=10, batch_size=32, epochs=150, experiment_name="interrupted_cl"),
]

with mp.Pool(3 * NUM_GPUS) as p:
    p.map(run_command, zip(commands, range(len(commands))))
