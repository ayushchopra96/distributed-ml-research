import torch
from torch import nn


class Clients(nn.Module):
    def __init__(self, client_models):
        super().__init__()
        self.client_models = nn.ModuleList(client_models)
        self.grad_from_servers = [None] * len(self.client_models)
        self.client_intermediates = [None] * len(self.client_models)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [None] * len(self.client_models)
        to_server = [None] * len(self.client_models)
        for i in range(len(self.client_models)):
            self.client_intermediates[i], outputs[i] = self.client_models[i](
                inputs[i])
            to_server[i] = self.client_intermediates[i].detach().requires_grad_()
        return to_server, outputs

    def client_backward(self, grads_from_server, selected_mask):
        self.grads_from_server = grads_from_server
        for i in range(len(self.client_models)):
            if not selected_mask[i]:
                assert(self.client_intermediates[i] is not None)
                assert(self.grads_from_server[i] is not None)

                self.client_intermediates[i].backward(
                    self.grads_from_server[i])

    def train(self):
        self.client_models.train()

    def eval(self):
        self.client_models.eval()
