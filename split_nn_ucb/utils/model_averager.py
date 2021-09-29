import torch
from torch import nn
from .comm_cost import compute_comm_cost

def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            
class ModelAverager:
    def __init__(self, model_fn):
        self.model_fn = model_fn

    @torch.no_grad()
    def average(self, split_nn):
        comm_cost = 0

        avg_model = self.model_fn().cuda()
        avg_model.apply(_initialize_weights)

        num_clients = len(split_nn.clients) 
        for k, m in split_nn.clients.items():
            for p, p_ in zip(
                avg_model.parameters(),
                m.parameters()
            ):
                p.add_(p_ / num_clients)
                comm_cost += compute_comm_cost(p_) * 2

        for k in split_nn.clients.keys():
            split_nn.clients[k].client_model.load_state_dict(avg_model.state_dict())
        return comm_cost