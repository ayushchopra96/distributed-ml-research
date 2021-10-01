import uuid
import optuna
from optuna.trial import TrialState
from uuid import uuid4
import os
import json

interrupt_ranges = [0.3, 0.45]#, 0.6, 0.75, 0.9]
interrupt_ranges = [f" --interrupt_range {item} " for item in interrupt_ranges]

l1_weights = [0., 0.1]#, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.000005]
l1_weights = [f" --l1_norm_weight {item} " for item in l1_weights]

num_combinations = len(l1_weights) * len(interrupt_ranges) * 2 * 2
num_trial_per_combination = 3
num_trials = num_trial_per_combination * num_combinations

def get_random_name():
    return str(uuid4())

def objective(trial):
    experiment_name = get_random_name()
    interrupt_range = trial.suggest_categorical("interrupt_range", interrupt_ranges)
    l1_weight = trial.suggest_categorical("l1_norm_weight", l1_weights)
    use_head = trial.suggest_categorical("use_head", ["", " --use_head "])
    use_additive = trial.suggest_categorical("use_additive", ["", " --use_additive "])
    
    command = f"python3 train.py --cifar --use_random --interrupted --num_clients 5 --k 3 --non_iid_50 --use_masked --use_lenet --use_contrastive --experiment_name grid_search_{experiment_name} --epochs 20 "
    command += interrupt_range
    command += l1_weight
    command += use_head
    command += use_additive

    os.system(command)   
    stats = json.load(open(f"./stats/grid_search_{experiment_name}.json", "r"))
    return max(eval(stats['Method Accuracy']))

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="hparam_non_iid_50",
        storage="sqlite:///hparam_non_iid_50.db",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=num_trials, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))