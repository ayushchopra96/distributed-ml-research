import torch
from torch import nn


class Server(nn.Module):
    def __init__(self, server_model, num_clients):
        super().__init__()
        self.server_model = server_model
        self.to_server = None
        self.grads_to_client = [None] * num_clients
        self.num_clients = num_clients

    def forward(self, to_server, selected_mask):
        self.to_server = to_server
        outputs = [None] * self.num_clients
        for i in range(self.num_clients):
            if not selected_mask[i]:
                assert(self.to_server[i] is not None)
                outputs[i] = self.server_model(self.to_server[i])
        return outputs

    def server_backward(self, selected_mask):
        for i in range(self.num_clients):
            if not selected_mask[i]:
                assert(self.grads_to_client[i] is not None)
                self.grads_to_client[i] = self.to_server.grad.clone()
        return self.grads_to_client

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()
