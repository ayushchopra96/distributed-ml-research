from re import M
import torch.nn as nn
import torch.nn.functional as F
import functools
import operator
from .layers import MaskedConv2d, MaskedLinear

class MaskedLeNet(nn.Module):
    def __init__(self, num_classes, hooked, num_clients, client_size_config=1, **kwargs):
        super().__init__()
        self.hooked = hooked
        self.client_size_config = client_size_config
        self.norm = nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0)
        if self.client_size_config <= 1:
            self.conv2 = MaskedConv2d(20, 50, kernel_size=5, num_masks=num_clients, padding=2)
            self.act_norm_pool = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0),
                nn.MaxPool2d(3, 2, padding=1)
            )
            self.fc1   = MaskedLinear(3200, 800, num_masks=num_clients)
            self.fc2   = MaskedLinear(800, 500, num_masks=num_clients)
            self.fc3   = MaskedLinear(500, num_classes, num_masks=num_clients)
        elif self.client_size_config <= 2:
            self.fc1   = MaskedLinear(3200, 800, num_masks=num_clients)
            self.fc2   = MaskedLinear(800, 500, num_masks=num_clients)
            self.fc3   = MaskedLinear(500, num_classes, num_masks=num_clients)
        elif self.client_size_config <= 3:
            self.fc2   = MaskedLinear(800, 500, num_masks=num_clients)
            self.fc3   = MaskedLinear(500, num_classes, num_masks=num_clients)
        elif self.client_size_config <= 4:
            self.fc3   = MaskedLinear(500, num_classes, num_masks=num_clients)

    def sparsity_constraint(self):
        loss = 0.
        for pname, p in self.named_parameters():
            if pname.find("masks") != -1:
                loss = loss + p.sigmoid().sum()
        return loss

    def forward(self, x, i):
        if self.client_size_config == 1:
            x = self.norm(x)
            x = self.conv2(i, x)
            x = self.act_norm_pool(x)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(i, x))
            x = F.relu(self.fc2(i, x))
            out = self.fc3(i, x)
        if self.client_size_config == 2:
            x = self.norm(x)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(i, x))
            x = F.relu(self.fc2(i, x))
            out = self.fc3(i, x)
        if self.client_size_config == 3:            
            x = F.relu(self.fc2(i, x))
            out = self.fc3(i, x)
        if self.client_size_config == 4:            
            out = self.fc3(i, x)
        return out

# class MaskedLeNet(nn.Module):
#     def __init__(self, num_classes, hooked, num_clients, emb_dim=None, **kwargs):
#         super(MaskedLeNet, self).__init__()
#         assert(hooked == False)
#         self.hooked = hooked
#         self.conv2 = MaskedConv2d(20, 50, kernel_size=5, num_masks=num_clients, padding=2)
#         self.act_norm_pool = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0),
#             nn.MaxPool2d((3, 3), (2, 2), padding=1)
#         )
#         self.fc1   = MaskedLinear(3200, 800, num_masks=num_clients)
#         self.fc2   = MaskedLinear(800, 500, num_masks=num_clients)
#         self.fc3   = MaskedLinear(500, num_classes, num_masks=num_clients)

#     def forward(self, x, i):
#         out = self.conv2(i, x)
#         out = self.act_norm_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(i, out)
#         out = self.fc2(i, out)
#         out = self.fc3(i, out)
#         return out
# 
#     def sparsity_constraint(self):
#         loss = 0.
#         for pname, p in self.named_parameters():
#             if pname.find("weight_masks") != -1:
#                 loss = loss + p.sigmoid().sum()
#         return loss

if __name__ == "__main__":
    import torch

    net2 = MaskedLeNet(10, False, 10, 1)
    num_params = 0
    for p in net2.parameters():
        num_params += p.numel()
    print("Num Params in MaskedLeNet: ", num_params)
    inp = torch.randn(32, 20, 16, 16)
    out2 = net2(inp, 7)
    print(out2.shape)
    