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

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, hooked=False, option='A', num_layers=1, emb_dim=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.num_layers = num_layers
        self.hooked = hooked
        if self.hooked:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option)
            self.conv_hook = nn.Conv2d(16, 64, kernel_size=1, bias=False)
            self.fc = nn.Linear(64, num_classes if emb_dim is None else emb_dim)
        else:
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option)
            self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option=option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.hooked:
            out = self.layer2(x)
            out = self.layer3(out)
            size = out.size(3)
            if not isinstance(size, int):
                size = size.item()
            out = F.avg_pool2d(out, size)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            layer1_out = self.layer1(out)
            out = self.conv_hook(layer1_out)
            size = out.size(3)
            if not isinstance(size, int):
                size = size.item()
            out = F.avg_pool2d(out, size)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return layer1_out, out


def resnet18(hooked=False, option='A', num_classes=10, emb_dim=None, num_layers=1):
    return ResNet(BasicBlock, [2, 2, 2], hooked=hooked, option=option, num_classes=num_classes, num_layers=num_layers, emb_dim=emb_dim)


def resnet32(hooked=False, option='A', num_classes=10, emb_dim=None, num_layers=1):
    return ResNet(BasicBlock, [5, 5, 5], hooked=hooked, option=option, num_classes=num_classes, num_layers=num_layers, emb_dim=emb_dim)


def resnet44(hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], hooked=hooked, option=option, num_classes=num_classes)


def resnet56(hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], hooked=hooked, option=option, num_classes=num_classes)


def resnet110(hooked=False, option='A', num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], hooked=hooked, option=option)


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

import torch.nn as nn
import torch.nn.functional as F
import functools
import operator

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(functools.reduce(operator.__add__,
                  [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

class LeNet(nn.Module):
    def __init__(self, num_classes, hooked, use_head=True, emb_dim=None):
        super(LeNet, self).__init__()
        self.hooked = hooked
        self.use_head = use_head
        if self.hooked:
            self.conv1 = nn.Sequential(
                Conv2dSamePadding(3, 20, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0),
                nn.MaxPool2d((3, 3), (2, 2), padding=1)
            )
            if self.use_head:
                self.conv_hook = nn.Conv2d(20, 128, kernel_size=1)
                self.head = nn.Linear(128, num_classes if emb_dim is None else emb_dim)
        else:
            self.conv2 = nn.Sequential(
                Conv2dSamePadding(20, 50, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0),
                nn.MaxPool2d(3, 2, padding=1)
            )
            self.fc1   = nn.Linear(3200, 800)
            self.fc2   = nn.Linear(800, 500)
            self.fc3   = nn.Linear(500, num_classes)

    def forward(self, x):
        if self.hooked:
            layer1_out = self.conv1(x)
            if self.use_head:
                emb = F.leaky_relu(self.conv_hook(layer1_out))
                size = emb.size(3)
                if not isinstance(size, int):
                    size = size.item()
                emb = F.max_pool2d(emb, size).reshape(emb.shape[0], -1)
                emb = self.head(emb)
                return layer1_out, emb
            else:
                return layer1_out, None
        else:
            out = self.conv2(x)
            out = out.view(out.shape[0], -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
            return out


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    net1 = LeNet(10, True, emb_dim=64, use_head=False)
    num_params = 0
    for p in net1.parameters():
        num_params += p.numel()

    inp = torch.randn(32, 3, 32, 32)
    feat_next, emb = net1(inp)
    print(feat_next.shape, emb)

    net2 = LeNet(10, False, 128)
    for p in net2.parameters():
        num_params += p.numel()
    print("Num Params in LeNet:", num_params)
    inp = torch.randn(*feat_next.shape)
    out2 = net2(inp)
    print(out2.shape)
    