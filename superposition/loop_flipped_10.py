from argparse import ArgumentParser
from dataclasses import dataclass
from models_cifar import resnet32
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from time import time
from fvcore.nn import FlopCountAnalysis
import random
from tqdm import trange
from torch.multiprocessing import Pool
import os
import numpy as np
import torch
from torch.functional import split
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from dataset_wrapper import ClassificationContrastiveDataset, collate_fn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from triplettorch import TripletDataset
from triplettorch import AllTripletMiner, HardNegativeTripletMiner
from split_nn import SplitNN, Clients, Server
from ucb import UCB

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def get_model(num_clients=100, interrupted=False, avg=False, cifar=True):
    if cifar:
        num_classes = 10
        option = 'A'
        T_max = 50 if interrupted else 200
    else:
        num_classes = 200
        option = 'B'
        T_max = 38 if interrupted else 140

    alice_models = []
    for i in range(num_clients):
        model_a = resnet32(hooked=True)
        alice_models.append(model_a)
    alice_models = Clients(alice_models)

    opt_list_alice = []
    scheduler_list_alice = []

    for i in range(num_clients):
        opt_list_alice.append(
            #             optim.SGD(alice.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            optim.Adam(
                alice_models.client_models[i].parameters(), lr=2e-3, weight_decay=1e-4)
        )
        scheduler_list_alice.append(
            CosineAnnealingLR(opt_list_alice[-1], T_max=T_max)
            #             ReduceLROnPlateau(opt_list_alice[-1], mode='max', factor=0.7, patience=5)
        )

    model_bob = resnet32(hooked=False)
#     opt_bob = optim.SGD(model_bob.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    opt_bob = optim.Adam(model_bob.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler_bob = CosineAnnealingLR(opt_bob, T_max=T_max)
#     scheduler_bob = ReduceLROnPlateau(opt_bob, mode='max', factor=0.7, patience=5)

    split_model = SplitNN(
        alice_models,
        Server(model_bob, num_clients),
        opt_list_alice,
        opt_bob,
        scheduler_list_alice,
        scheduler_bob,
        triplet_loss=HardNegativeTripletMiner(0.5).to(device),
        clf_loss=nn.CrossEntropyLoss(reduction='none'),
        interrupted=interrupted,
        avg=avg,
    )

    return split_model


def to_interrupt_or_not(use_vanilla, use_ucb, selected_ids, i, interrupt_range, ep):
    ucb_interrupt = False

    in_interrupt_range = False
    if ep in range(*interrupt_range):
        in_interrupt_range = True

    if i not in selected_ids:
        ucb_interrupt = True

    return not use_vanilla or ((ucb_interrupt and use_ucb) or in_interrupt_range)


def get_mask(num_clients, use_vanilla, use_ucb, selected_ids, interrupt_range, ep):
    mask = torch.zeros((num_clients, ))
    for i in range(num_clients):
        mask[i] = to_interrupt_or_not(
            use_vanilla, use_ucb, selected_ids, i, interrupt_range, ep)
    return mask.bool().to(device)


def experiment_ucb(
    experiment_name,
    split_nn,
    train_dataloader,
    test_loader,
    interrupt_range,
    epochs,
    k,
    use_ucb,
    use_vanilla,
    use_contrastive,
    discount_hparam,
    dataset_sizes,
    poll_clients,
    steps=None,
    num_clients=100,
):

    flops_split, steps = 0, 0
    (
        flops_split_list,
        acc_split_list,
        steps_list,
        time_list,
    ) = ([], [], [], [])

    bandit = UCB(num_clients, discount_hparam, dataset_sizes, k)

    use_contrastive = 1 if use_contrastive else 0

    flag = True
    t = trange(epochs, desc="", leave=True)
    split_nn.to(device)

    selected_ids = random.sample(list(range(num_clients)), k)

    wallclock_start = time()

    for ep in t:  # 200
        for x, y, xc, yc in train_dataloader:
            mask = get_mask(num_clients, use_vanilla, use_ucb,
                            selected_ids, interrupt_range, ep)
            # zero grads
            split_nn.zero_grads()
            split_nn.train()

            x, y = x.transpose(0, 1), y.transpose(0, 1)
            x, y = x.to(device), y.to(device)

            xc, yc = xc.transpose(0, 1), yc.transpose(0, 1)
            xc, yc = xc.to(device), yc.to(device)

            triplet_loss, stump_clf_loss, final_clf_loss, flops = split_nn(
                x, y, xc, yc, mask, flops_split, use_contrastive)

            mask = mask.unsqueeze(-1).expand(y.shape)

            not_selected_loss = (use_contrastive * triplet_loss +
                                 (1 - use_contrastive) * stump_clf_loss) * mask
            selected_loss = final_clf_loss * (~mask)
            losses = selected_loss + not_selected_loss

            loss_mean, loss_std = losses.mean(
                -1).detach(), losses.std(-1).detach()
            if not poll_clients:
                loss_mean, loss_std = loss_mean * \
                    (~mask[:, 0]), loss_std * (~mask[:, 0])

            losses.mean().backward()
            split_nn.backward(mask[:, 0])
            split_nn.step()

            bandit.update_clients(loss_mean, loss_std, mask[:, 0])

            selected_ids = bandit.select_clients()
            bandit.end_round()

            steps += 1

        # do eval and record after epoch
        acc_split, alice_split = split_nn.evaluator(test_loader)

        # trigger scheduler bob after a warmup of 10 epochs
        if ep >= 10:
            if ep not in range(*interrupt_range):
                out = True
                split_nn.scheduler_step(
                    acc_split, out=out, step_bob=True)

        acc_split_list.append(acc_split)
        steps_list.append(steps)
        time_list.append(time() - wallclock_start)
        flops_split_list.append(flops_split)

        t.set_description(
            f"Split: {acc_split}, Steps: {steps}", refresh=True
        )

        if ep % 50 == 0 and ep > 0:
            if not os.path.isdir(f"runs/{experiment_name}/"):
                os.makedirs(f"runs/{experiment_name}/", exist_ok=True)
            # torch.save(split_nn.state_dict(),
            #            f"runs/{experiment_name}/split_nn_{steps}")

    return (
        acc_split_list,
        acc_interrupted_list,
        flops_split_list,
        flops_interrupted_list,
        time_list,
        steps_list,
    )


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


class RandomGammaCorrection(object):

    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):

        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0, 0, 0, 0.90, 1.04, 1.08]
            self.gamma = random.choice(gammas)

#         print(self.gamma)
        if self.gamma == 0:
            return image

        else:
            return transforms.functional.adjust_gamma(image, self.gamma, gain=1)


@dataclass
class hparam:
    cifar: bool = True
    num_clients: int = 10
    k: int = 2
    discount: float = 0.7
    poll_clients: bool = False
    interrupted: bool = True  # Interruption OFF/ON
    batch_size: int = 32
    epochs: int = 150
    use_ucb: bool = True
    use_contrastive: bool = True
    use_vanilla: bool = True


if __name__ == "__main__":
    parser = ArgumentParser()
    for k, v in hparam().__dict__.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", default=False, action='store_true')
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    hparams_ = parser.parse_args()
    print(hparams_)
    temp = []
    for k, v in hparams_.__dict__.items():
        temp.append(f"{k}_{v}")

    experiment_name = "-".join(temp)
    print(experiment_name)

    cifar = hparams_.cifar
    num_clients = hparams_.num_clients
    k = hparams_.k
    discount = hparams_.discount
    poll_clients = hparams_.poll_clients
    interrupted = hparams_.interrupted  # Interruption OFF/ON
    batch_size = hparams_.batch_size
    epochs = hparams_.epochs

    if cifar:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_val
        )

    else:
        train_dir = "./data/tiny-imagenet-200/train/"
        val_dir = "./data/tiny-imagenet-200/val/"
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        __imagenet_pca = {'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
                          'eigvec': torch.Tensor([[-0.5675,  0.7192,  0.4009],
                                                  [-0.5808, -0.0045, -0.8140],
                                                  [-0.5836, -0.6948,  0.4203],
                                                  ])
                          }
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(.4, .4, .4),
                transforms.RandomAffine(degrees=(-5, 5), scale=(0.9, 1.08)),
                RandomGammaCorrection(),
                transforms.ToTensor(),
                Lighting(0.1, __imagenet_pca['eigval'],
                         __imagenet_pca['eigvec']),
                norm,
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                norm,
            ]
        )

        trainset = dsets.ImageFolder(train_dir, transform=transform)
        testset = dsets.ImageFolder(val_dir, transform=transform_val)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=not cifar
    )

    train_dataset = ClassificationContrastiveDataset(
        num_clients, trainset, hparams_.use_contrastive, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=not cifar, collate_fn=collate_fn
    )

    if interrupted:
        interrupt_range = [0, int(0.75*epochs)]
    else:
        interrupt_range = [-2, 0]  # Hack for not using Local Parallelism

    split_nn = get_model(num_clients=num_clients,
                         interrupted=hparams_.use_vanilla, cifar=cifar)
    (
        acc_split_list,
        acc_interrupted_list,
        flops_split_list,
        flops_interrupted_list,
        time_list,
        steps_list,
    ) = experiment_ucb(
        experiment_name,
        split_nn,
        train_dataloader,
        test_loader,
        k=hparams_.k,
        use_contrastive=hparams_.use_contrastive,
        use_vanilla=hparams_.use_vanilla,
        use_ucb=hparams_.use_ucb,
        poll_clients=poll_clients,
        discount_hparam=hparams_.discount,
        dataset_sizes=train_dataset.train_sizes,
        interrupt_range=interrupt_range,
        epochs=epochs,
        num_clients=num_clients,
    )

    import json

    out_dict = {
        "Method Accuracy": acc_split_list,
        "Flops": flops_split_list,
        "Time": time_list,
        "Steps": steps_list,
        "hparams": hparams_.__dict__
    }
    print(out_dict)

    out_dict = {
        "Method Accuracy": str(acc_split_list),
        "Flops": str(flops_split_list),
        "Time": str(time_list),
        "Steps": steps_list,
        "hparams": hparams_.__dict__
    }

    with open(f"stats/{experiment_name}.json", "w") as f:
        json.dump(out_dict, f)

    # split_nn = get_model(num_clients=num_clients, interrupted=False, avg=True)
    # interrupted_nn = get_model(num_clients=num_clients, interrupted=True, avg=True)

    # (
    #     acc_split_list,
    #     acc_interrupted_list,
    #     flops_split_list,
    #     flops_interrupted_list,
    #     time_list,
    #     steps_list,
    # ) = experiment_avg_grad(
    #     split_nn,
    #     interrupted_nn,
    #     cifar_train_loader_list,
    #     cifar_test_loader_list,
    #     interrupt_range=interrupt_range,
    #     epochs=5000,
    #     num_clients=num_clients,
    # )

    # out_dict = {
    #     "SplitNN Accuracy": acc_split_list,
    #     "Interrupted Accuracy": acc_interrupted_list,
    #     "Flops SplitNN": flops_split_list,
    #     "Flops Interrupted": flops_interrupted_list,
    #     "Time": time_list,
    # }
    # print(out_dict)

    # out_dict = {
    #     "SplitNN Accuracy": str(acc_split_list),
    #     "Interrupted Accuracy": str(acc_interrupted_list),
    #     "Flops SplitNN": str(flops_split_list),
    #     "Flops Interrupted": str(flops_interrupted_list),
    #     "Time": str(time_list),
    # }

    # with open("stats_resnet_averaged.json", "w") as f:
    #     json.dump(out_dict, f)
