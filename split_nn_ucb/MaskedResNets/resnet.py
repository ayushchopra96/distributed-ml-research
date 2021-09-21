'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from .layers import MaskedLinear, MaskedBatchNorm2d, MaskedConv2d
__all__ = ['ResNet', 'resnet20', 'resnet32',
           'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, MaskedLinear) or isinstance(m, MaskedConv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, num_masks, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.num_masks = num_masks
        self.conv1 = MaskedConv2d(
            in_planes, planes, num_masks, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm2d(planes, num_masks)
        self.conv2 = MaskedConv2d(
            planes, planes, num_masks, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MaskedBatchNorm2d(planes, num_masks)
        self.option = option
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MaskedConv2d(in_planes, self.expansion * planes,
                                      num_masks, kernel_size=1, stride=stride, bias=False),
                    MaskedBatchNorm2d(
                        self.expansion * planes, num_masks)
                )

    def forward(self, args):
        x, i = args
        out = F.relu(self.bn1(i, self.conv1(i, x)))
        out = self.bn2(i, self.conv2(i, out))
        if self.shortcut is not None:
            if self.option == "A":
                out = out + self.shortcut(x)
            else:
                out = out + self.shortcut(i, x)
        out = F.relu(out)
        return out, i 


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_masks, num_clients, num_classes=10, hooked=False, option='A'):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.num_masks = num_masks
        self.hooked = hooked
        assert(hooked == False)

        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], num_masks, stride=2, option=option)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], num_masks, stride=2, option=option)
        self.linear = MaskedLinear(64, num_classes, num_masks)

        self.apply(_weights_init)

    def sparsity_constraint(self):
        loss = 0.
        for pname, p in self.named_parameters():
            if pname.find("weight_masks") != -1:
                loss = loss + p.sigmoid().sum()
        return loss

    def _make_layer(self, block, planes, num_blocks, num_masks, stride, option):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                                num_masks, stride, option=option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, i):
        out, _ = self.layer2((x, i))
        out, _ = self.layer3((out, i))
        size = out.size(3)
        if not isinstance(size, int):
            size = size.item()
        out = F.avg_pool2d(out, size)
        out = out.view(out.size(0), -1)
        out = self.linear(i, out)
        return out


def resnet18(num_masks, num_clients, hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2], num_masks, num_clients, hooked=hooked, option=option, num_classes=num_classes)


def resnet32(num_masks, num_clients, hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_masks, num_clients, hooked=hooked, option=option, num_classes=num_classes)


def resnet44(num_masks, num_clients, hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_masks, num_clients, hooked=hooked, option=option, num_classes=num_classes)


def resnet56(num_masks, num_clients, hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_masks, num_clients, hooked=hooked, option=option, num_classes=num_classes)


def resnet110(num_masks, num_clients, hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_masks, num_clients, hooked=hooked, option=option)


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    # for net_name in __all__:
    #     if net_name.startswith('resnet'):
    #         print(net_name)
    #         test(globals()[net_name]())
    #         print()

    net = resnet32(3, 10, hooked=True)

    inp = torch.randn(32, 3, 32, 32)

    out = net(inp)
    print(out.shape)
