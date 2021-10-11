from MaskedResNets import MaskedResNet, resnet32
import torch

model = resnet32(3, 10, hooked=False, num_classes=10)

alpha = 5e-5
def sparsity_constraint(self):
    loss = 0.
    for pname, p in self.named_parameters():
        if pname.find("masks") != -1:
            loss = loss + torch.norm(p, p=1)
    return loss


loss = sparsity_constraint(model) * alpha
print(loss)
