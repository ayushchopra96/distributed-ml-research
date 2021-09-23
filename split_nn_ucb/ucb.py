import torch
from einops import repeat
import random
import numpy as np
from numba import njit
from vowpalwabbit import pyvw

class VWBandit:
    def __init__(self, num_clients, discount_hparam, dataset_sizes, k):
        self.model = pyvw.vw(f"--cb_explore {num_clients} --epsilon 0.05", quiet=True)
        self.losses_mean, self.losses_std, self.selection_mask = None, None, None
        self.discount = discount_hparam
        self.num_clients = num_clients
        self.k = k
        self.max_history = 16  # store data for only last this number of steps
        self.end_round()

    def update_client(self, client_id, loss_mean, loss_std, was_selected):
        round_counter = len(self.losses_mean) - 1
        if was_selected:
            self.losses_mean[round_counter][client_id] = loss_mean
            self.losses_std[round_counter][client_id] = loss_std
        # print(self.losses_mean[self.round_counter][client_id], self.losses_std[self.round_counter][client_id])
        self.selection_mask[round_counter][client_id] = 1. if was_selected else 0.
    
    def train(self):
        upper_bound = self.losses_mean[-1] * self.selection_mask[-1] + 1.96 * self.losses_std[-1] / np.sqrt(self.selection_mask[-1].sum(0) + 1e-4) * self.selection_mask[-1] 
        actions, probs = self.label
        for a, p in zip(actions, probs):
            reward = upper_bound[a]
            example = f"{a}:{reward}:{p} | {self.prev_features}"
            self.model.learn(example)
        
    def end_round(self):
        if self.losses_std is not None:
            self.train()

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

        if round_counter >= self.max_history:
            self.losses_mean = np.delete(self.losses_mean, 0, 0)
            self.losses_std = np.delete(self.losses_std, 0, 0)
            self.selection_mask = np.delete(self.selection_mask, 0, 0)

    def make_features(self):
        example = ""
        feature_counter = 0
        # Losses mean
        for r in range(self.losses_mean.shape[0]):
            for c in range(self.losses_mean[0].shape[0]):
                example += f" feature{feature_counter}={float(self.losses_mean[r, c])}"
                feature_counter += 1

        # Losses std
        for r in range(self.losses_mean.shape[0]):
            for c in range(self.losses_mean[0].shape[0]):
                example += f" feature{feature_counter}={float(self.losses_std[r, c])}"
                feature_counter += 1

        # selection mask
        for r in range(self.losses_mean.shape[0]):
            for c in range(self.losses_mean[0].shape[0]):
                example += f" feature{feature_counter}={float(self.selection_mask[r, c])}"
                feature_counter += 1

        self.prev_features = example
        return example

    def select_clients(self):
        self.make_features()
        pmf = self.model.predict(self.prev_features)
        actions, probs = self.get_actions(pmf)
        self.label = (actions, probs)
        return actions

    def get_actions(self, pmf):
        pmf = np.array(pmf)
        # print(pmf)
        idx = np.argsort(pmf)[-self.k:]
        prob = pmf[idx].tolist()
        return idx.tolist(), prob

class UniformRandom:
    def __init__(self, num_clients, discount_hparam, dataset_sizes, k):
        pass
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
        # scores2 = BayesianAdvantageVec(
        #     self.discount, 
        #     self.num_clients, 
        #     self.losses_mean, 
        #     self.losses_std, 
        #     self.selection_mask
        # )
        # assert(np.allclose(scores, scores2))
        # print(np.allclose(scores, scores2))
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
            mean[c] += (discount ** (round_counter - 1 - t)) * losses_mean[t][c]
            std[c] += (discount ** (round_counter - 1 - t)) * losses_std[t][c]
            counts[c] += (discount ** (round_counter - 1 - t)) * (client_selection_mask[t][c])
    scores = mean + 2 * std / np.sqrt(counts)
    return scores

def BayesianAdvantageVec(
    discount, 
    num_clients, 
    losses_mean, 
    losses_std, 
    client_selection_mask
    ):
    round_counter = losses_mean.shape[0] - 1 - np.arange(0, losses_mean.shape[0])
    round_counter = round_counter.reshape(1, -1)
    print(round_counter)
    mean = np.power(discount, round_counter) @ losses_mean
    std = np.power(discount, round_counter) @ losses_std
    counts = np.power(discount, round_counter) @ client_selection_mask
    scores = mean + 2 * std / np.sqrt(counts)
    return scores

def dataset_fractions(dataset_sizes):
    return dataset_sizes / sum(dataset_sizes)


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
    from joblib import Parallel, delayed
    from tqdm import trange
    num_rounds = 150
    num_clients = 10
    
    torch.manual_seed(0)
    np.random.seed(0)

    losses = torch.randn((num_rounds, num_clients))
    std = torch.randn((num_rounds, num_clients))

    mask = torch.randint_like(losses, 0, 2)
    # mask = torch.zeros_like(losses)
    dataset_sizes = torch.randint(100, 10000, (num_clients,))

    p = dataset_fractions(dataset_sizes)

    def _run_one(r, c, selected, selected_r, b):
        cum_reward = 0
        cum_reward_r = 0

        l = torch.rand((num_clients,)).uniform_((c) / (r+1), (c+1) / (r+1))
        if selected_r:
            cum_reward_r += l.mean().item()
        if selected:
            b.update_client(c, l.mean(), l.std(), selected)
            cum_reward += l.mean().item()
        else:
            # l = torch.rand((num_clients,)).uniform_((10 * c) / (r+1), 10 * (c+1) / (r+1))
            b.update_client(c, None, None, selected)
        return cum_reward, cum_reward_r

    b_r = UniformRandom(
        num_clients,
        0.9,
        dataset_sizes,
        6
    )
    
    # selected_ids = random.sample(list(range(num_clients)), 6)
    # # print(Parallel(n_jobs=-1)(delayed(_run_one)(0, c, c in selected_ids, b) for c in range(num_clients)))
    # random_rewards = []
    # cum_reward = 0.
    # for r in trange(num_rounds):
    #     for _ in range(160):
    #         cum_reward += sum(Parallel(n_jobs=-1, require='sharedmem')(delayed(_run_one)(r, c, c in selected_ids, b) for c in range(num_clients)))
    #         b.end_round()
    #         selected_ids = b.select_clients()

    #         random_rewards.append(cum_reward)
    
    b = VWBandit(
        num_clients,
        0.9,
        dataset_sizes,
        6
    )
    
    selected_ids = b.select_clients()
    selected_ids_r = b_r.select_clients()
    vw_rewards = []
    cum_reward = 0.
    cum_reward_r = 0.
    for r in trange(num_rounds):
        for _ in range(160):
            out = Parallel(n_jobs=-1, require='sharedmem')(delayed(_run_one)(r, c, c in selected_ids, c in selected_ids_r, b) for c in range(num_clients))
            cum_reward += sum([item[0] for item in out])
            cum_reward_r += sum([item[1] for item in out])

            b.end_round()

            selected_ids = b.select_clients()
            selected_ids_r = b_r.select_clients()

            vw_rewards.append(cum_reward - cum_reward_r)

    # b = BayesianUCB(
    #     num_clients,
    #     0.9,
    #     dataset_sizes,
    #     6
    # )
    # selected_ids = random.sample(list(range(num_clients)), 6)
    # ucb9_rewards = []
    # cum_reward = 0.
    # for r in trange(num_rounds):
    #     for _ in range(160):
    #         cum_reward += sum(Parallel(n_jobs=-1)(delayed(_run_one)(r, c, c in selected_ids, b) for c in range(num_clients)))
    #         b.end_round()
    #         selected_ids = b.select_clients()

    #         ucb9_rewards.append(cum_reward)

    # b = BayesianUCB(
    #     num_clients,
    #     0.8,
    #     dataset_sizes,
    #     6
    # )
    # selected_ids = random.sample(list(range(num_clients)), 6)
    # ucb8_rewards = []
    # cum_reward = 0.
    # for r in trange(num_rounds):
    #     for _ in range(160):
    #         cum_reward += sum(Parallel(n_jobs=-1, require='sharedmem')(delayed(_run_one)(r, c, c in selected_ids, b) for c in range(num_clients)))
    #         b.end_round()
    #         selected_ids = b.select_clients()

    #         ucb8_rewards.append(cum_reward)

    # b = BayesianUCB(
    #     num_clients,
    #     0.6,
    #     dataset_sizes,
    #     6
    # )
    # selected_ids = random.sample(list(range(num_clients)), 6)
    # ucb6_rewards = []
    # cum_reward = 0.
    # for r in trange(num_rounds):
    #     for _ in range(160):
    #         cum_reward += sum(Parallel(n_jobs=-1, require='sharedmem')(delayed(_run_one)(r, c, c in selected_ids, b) for c in range(num_clients)))
    #         b.end_round()
    #         selected_ids = b.select_clients()

    #         ucb6_rewards.append(cum_reward)

    # b = BayesianUCB(
    #     num_clients,
    #     0.5,
    #     dataset_sizes,
    #     6
    # )
    # selected_ids = random.sample(list(range(num_clients)), 6)
    # ucb5_rewards = []
    # cum_reward = 0.
    # for r in trange(num_rounds):
    #     for _ in range(160):
    #         cum_reward += sum(Parallel(n_jobs=-1, require='sharedmem')(delayed(_run_one)(r, c, c in selected_ids, b) for c in range(num_clients)))
    #         b.end_round()
    #         selected_ids = b.select_clients()

    #         ucb5_rewards.append(cum_reward)

    # b = BayesianUCB(
    #     num_clients,
    #     0.4,
    #     dataset_sizes,
    #     6
    # )
    # selected_ids = random.sample(list(range(num_clients)), 6)
    # ucb4_rewards = []
    # cum_reward = 0.
    # for r in trange(num_rounds):
    #     for _ in range(160):
    #         cum_reward += sum(Parallel(n_jobs=-1, require='sharedmem')(delayed(_run_one)(r, c, c in selected_ids, b) for c in range(num_clients)))
    #         b.end_round()
    #         selected_ids = b.select_clients()

    #         ucb4_rewards.append(cum_reward)

    # b = BayesianUCB(
    #     num_clients,
    #     0.95,
    #     dataset_sizes,
    #     6
    # )
    # selected_ids = random.sample(list(range(num_clients)), 6)
    # ucb95_rewards = []
    # cum_reward = 0.
    # for r in trange(num_rounds):
    #     for _ in range(160):
    #         cum_reward += sum(Parallel(n_jobs=-1, require='sharedmem')(delayed(_run_one)(r, c, c in selected_ids, b) for c in range(num_clients)))
    #         b.end_round()
    #         selected_ids = b.select_clients()

    #         ucb95_rewards.append(cum_reward)

    # import plotly.graph_objects as go

    # fig = go.Figure()
    # fig = fig.add_trace(
    #     go.Scatter(y=random_rewards, name='Uniform Random')
    # )
    # fig = fig.add_trace(
    #     go.Scatter(y=ucb9_rewards, name='Bayesian UCB gamma=0.9')
    # )
    # fig = fig.add_trace(
    #     go.Scatter(y=ucb8_rewards, name='Bayesian UCB gamma=0.8')
    # )
    # fig = fig.add_trace(
    #     go.Scatter(y=ucb4_rewards, name='Bayesian UCB gamma=0.4')
    # )
    # fig = fig.add_trace(
    #     go.Scatter(y=ucb5_rewards, name='Bayesian UCB gamma=0.5')
    # )
    # fig = fig.add_trace(
    #     go.Scatter(y=ucb6_rewards, name='Bayesian UCB gamma=0.6')
    # )
    # fig = fig.add_trace(
    #     go.Scatter(y=ucb95_rewards, name='Bayesian UCB gamma=0.95')
    # )

    # fig.show()
    
    import plotly.graph_objects as go

    fig = go.Figure()
    fig = fig.add_trace(
        go.Scatter(y=vw_rewards, name='VowpalWabbit')
    )    
    fig.show()