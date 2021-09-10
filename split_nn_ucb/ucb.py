import torch
from einops import repeat
import random
import numpy as np
from numba import njit

class UniformRandom:
    def __init__(self, num_clients, discount_hparam, dataset_sizes, k):
        self.num_clients = num_clients
        self.k = k

    def select_clients(self):
        return random.sample(list(range(self.num_clients)), self.k)

    def end_round(self):
        pass

    def update_client(self, *args, **kwargs):
        pass

class BayesianUCB:
    def __init__(self, num_clients, discount_hparam, dataset_sizes, k):
        self.losses_mean, self.losses_std, self.selection_mask = None, None, None
        self.discount = discount_hparam
        self.dataset_ratio = dataset_fractions(dataset_sizes)
        self.num_clients = num_clients
        self.k = k
        self.max_history = 2048  # store data for only last this number of steps
        self.end_round()
        
    def update_client(self, client_id, loss_mean, loss_std, was_selected):
        round_counter = len(self.losses_mean) - 1
        if was_selected:
            self.losses_mean[round_counter][client_id] = loss_mean
            self.losses_std[round_counter][client_id] = loss_std
        # print(self.losses_mean[self.round_counter][client_id], self.losses_std[self.round_counter][client_id])
        self.selection_mask[round_counter][client_id] = 1. if was_selected else 0.
    

    def end_round(self):
        zeros = np.zeros((1, self.num_clients))
        if self.losses_std is not None:
            self.losses_mean = np.append(self.losses_mean, zeros.copy(), axis=0)
            self.losses_std = np.append(self.losses_std, zeros.copy(), axis=0)
            self.selection_mask = np.append(self.selection_mask, zeros.copy(), axis=0)
        else:
            self.losses_mean = zeros.copy()
            self.losses_std =  zeros.copy()
            self.selection_mask = zeros.copy()

        round_counter = len(self.losses_mean)

        if round_counter == 1:
            self.losses_mean[0] = 100.
            self.losses_std[0] = 100.
        elif round_counter == 2:
            self.losses_mean[1, :] = (100. + self.losses_mean[0, :]) / 2 
            self.losses_std[1, :] = (100. + self.losses_std[0, :]) / 2
        else:
            self.losses_mean[-1, :] = (self.losses_mean[-3, :] + self.losses_mean[-2, :]) / 2
            self.losses_std[-1, :] = (self.losses_std[-3, :] + self.losses_std[-2, :]) / 2
        # print(self.losses_mean, len(self.losses_mean))
        # assert(all(self.losses_mean[-1]))
        # assert(all(self.losses_std[-1]))


        if round_counter >= self.max_history:
            self.losses_mean = np.delete(self.losses_mean, 0, 0)
            self.losses_std = np.delete(self.losses_std, 0, 0)
            self.selection_mask = np.delete(self.selection_mask, 0, 0)
    
    def select_clients(self):
        scores = BayesianAdvantage(
            self.discount, 
            self.num_clients, 
            self.losses_mean, 
            self.losses_std, 
            self.selection_mask
        )
        # print(scores)
        return topk(list(scores), self.k)

@njit()
def BayesianAdvantage(
    discount, 
    num_clients, 
    losses_mean, 
    losses_std, 
    client_selection_mask
    ):
    round_counter = losses_mean.shape[0]
    mean = np.zeros(num_clients)
    std = np.zeros(num_clients)
    counts = np.zeros(num_clients) + 1e-15
    for t in range(round_counter):
        for c in range(num_clients):
            mean[c] += discount ** (round_counter - 1 - t) * losses_mean[t][c]
            std[c] += discount ** (round_counter - 1 - t) * losses_std[t][c]
            counts[c] += discount ** (t) * (1 - client_selection_mask[t][c])
    # print("mask", client_selection_mask.sum(0))
    # print("losses", losses_mean.sum(0))
    # print("std", losses_std.sum(0))
    # print(mean, counts, std)
    scores = mean / counts + 2 * std / np.sqrt(counts)
    return scores

class UCB:
    def __init__(self, num_clients, discount_hparam, dataset_sizes, k):
        self.losses_mean, self.losses_std, self.selection_mask = [], [], []
        self.discount = discount_hparam
        self.dataset_ratio = dataset_fractions(dataset_sizes)
        self.num_clients = num_clients
        self.k = k
        self.round_counter = -1
        self.max_history = 2048  # store data for only last this number of steps
        self.end_round()

    def update_client(self, client_id, loss_mean, loss_std, was_selected):
        if not was_selected or (loss_mean is None and loss_std is None):
            # loss_mean and loss_std are None if the client wasn't selected
            if self.round_counter == 0:
                # If it is the initial round
                loss_mean = 0.
                loss_std = 100.  # Assume large std
            else:
                # Assume loss statistics are discounted previous round's statistics
                loss_mean = self.losses_mean[self.round_counter -
                                             1][client_id] * self.discount
                loss_std = self.losses_std[self.round_counter -
                                            1][client_id] * self.discount

        self.losses_mean[self.round_counter][client_id] = loss_mean
        self.losses_std[self.round_counter][client_id] = loss_std
        self.selection_mask[self.round_counter][client_id] = 1. if was_selected else 0.

    def end_round(self):
        zeros = [0.] * self.num_clients
        self.losses_mean.append(zeros)
        self.losses_std.append(zeros)
        self.selection_mask.append(zeros)
        self.round_counter += 1

        if self.round_counter >= self.max_history:
            self.losses_mean.pop(0)
            self.losses_std.pop(0)
            self.selection_mask.pop(0)
            self.round_counter -= 1

    def select_clients(self):
        A = advantage(
            torch.tensor(self.losses_mean),
            torch.tensor(self.selection_mask),
            self.discount,
            torch.tensor(self.losses_std),
            self.dataset_ratio
        )
        return select_clients(A, self.k)


def advantage(
    client_losses,
    client_selection_mask,
    discount_hparam,
    client_loss_stds,
    dataset_ratio
):
    # discount_hparam decides how much to weigh previous round's information.
    # discount_hparam can be in [0, 1]. If equal to 1 -> don't use any old information
    # If equal to 0 -> weigh old information equally to new information

    # client_losses -> (num_rounds, num_clients)

    # client_loss_stds -> (num_rounds, num_clients)
    num_rounds = client_losses.shape[0]

    # max_std is the maximum standard deviation in the local loss computed over the latest update
    # Compute max_std for last round
    max_std = client_loss_stds[-1, :].max() + 1e-15

    # Compute discounted time indices
    gamma = torch.pow(
        discount_hparam,
        torch.arange(num_rounds-1, -1, -1)
    ).unsqueeze(1)

    L = loss_term(
        client_losses,
        client_selection_mask,
        discount_hparam,
        gamma
    )

    N, t = client_selection_term(
        client_selection_mask,
        gamma
    )

    U = exploration_term(
        t,
        N,
        max_std
    )

    p = dataset_ratio

    exploitation = L / N * p
    exploration = U * p
    return exploitation + exploration


def loss_term(
    client_losses,
    client_selection_mask,
    discount_hparam,
    gamma
):

    estimated_losses = client_losses.clone()
    num_rounds, num_clients = client_losses.shape
    for r in range(1, num_rounds):
        # Assuming the loss in the next round is the average of losses in the current round and previous round
        estimated_losses[r, :] = (
            estimated_losses[r, :] + client_losses[r - 1, :]) / 2

    # Sum over all rounds
    return (
        repeat(gamma, 'r c -> r (repeat c)', repeat=num_clients) *
        client_selection_mask * estimated_losses).sum(dim=0)


def dataset_fractions(dataset_sizes):
    return dataset_sizes / sum(dataset_sizes)


def client_selection_term(client_selection_mask, gamma):
    # client_selection_mask -> (num_rounds, num_clients)
    n = (gamma * client_selection_mask).sum(dim=0) + 1e-15
    t = gamma.sum()
    return n, t


def exploration_term(t, client_selection_terms, max_std):
    return torch.sqrt(
        2 * max_std * max_std * torch.log(t) / client_selection_terms
    )


def select_clients(A, k):
    return topk(
        list(A.numpy()),
        k
    )


def topk(arr: list, k: int) -> list:
    # Non-deterministic top k
    # Uses some hacks to break ties randomly

    dummy = list(range(len(arr)))
    random.shuffle(dummy)
    idx = range(len(arr))
    tuples = list(zip(arr, dummy, idx))
    tuples.sort(reverse=True)
    return [item[-1] for item in tuples[:k]]


if __name__ == "__main__":
    num_rounds = 150
    num_clients = 10

    losses = torch.randn((num_rounds, num_clients))
    std = torch.randn((num_rounds, num_clients))

    mask = torch.randint_like(losses, 0, 2)
    # mask = torch.zeros_like(losses)
    dataset_sizes = torch.randint(100, 10000, (num_clients,))

    p = dataset_fractions(dataset_sizes)
    A = advantage(losses, mask, 0.97, std, p)
    assert(A.shape == p.shape)

    print(A)

    print(select_clients(A, 3))

    b = UCB(
        num_clients,
        0.7,
        dataset_sizes,
        3
    )
    b = BayesianUCB(
        num_clients,
        0.85,
        dataset_sizes,
        3
    )
    selected_ids = random.sample(list(range(num_clients)), 3)
    for r in range(num_rounds):
        for _ in range(160):
            for c in range(num_clients):
                l = torch.rand((dataset_sizes[0],)).uniform_((c) / (r+1), (c+1) / (r+1))# * (c + 1) * (1 / (r + 1))
                # l = torch.zeros((dataset_sizes[0],))
                # print(c in selected_ids)
                b.update_client(c, l.mean(), l.std(), c in selected_ids)
        b.end_round()
        selected_ids = b.select_clients()
        print(selected_ids)
