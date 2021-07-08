import torch
from einops import repeat
import random

class UCB:
    def __init__(self, num_clients, discount_hparam, dataset_sizes, k):
        self.losses_mean, self.losses_std, self.selection_mask = [], [], []
        self.discount = discount_hparam
        self.dataset_ratio = dataset_fractions(dataset_sizes)
        self.num_clients = num_clients
        self.k = k
        self.round_counter = -1
        self.max_history = 2048 # store data for only last 10000 steps
        self.end_round()

    def update_client(self, client_id, losses, was_selected):
        if isinstance(losses, torch.FloatTensor):
            losses = losses.clone()
        else:
            losses = torch.tensor(losses)

        mean, std = torch.mean(losses), torch.std(losses)
        self.losses_mean[self.round_counter][client_id] = mean
        self.losses_std[self.round_counter][client_id] = std
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
    # discount_hparam can be in [0, 1]. If equal to 0 -> don't use any old information
    # If equal to 1 -> weigh old information equally to new information

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
    A = advantage(losses, mask, 0.7, std, p)
    assert(A.shape == p.shape)

    print(A)

    print(select_clients(A, 3))

    b = UCB(
        num_clients,
        0.7,
        dataset_sizes,
        3
    )
    selected_ids = random.sample(list(range(num_clients)), 3)
    for r in range(num_rounds):
        for c in range(num_clients):
            l = torch.randn((dataset_sizes[0],))
            # l = torch.zeros((dataset_sizes[0],))
            # print(c in selected_ids)
            b.update_client(c, l, c in selected_ids)
        selected_ids = b.select_clients()
        b.end_round()
        print(selected_ids)
