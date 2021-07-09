from ucb import UCB
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import h5py
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
from dataset_wrapper import DataWrapper
import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)
from models_cifar import resnet32

# from tqdm.notebook import


# from tqdm.notebook import tqdm

# from torch.utils.tensorboard import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# writer = SummaryWriter()


class Client(nn.Module):
    def __init__(self, client_model):
        super().__init__()
        self.client_model = client_model
        self.grad_from_server = None
        self.client_intermediate = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.client_intermediate, output = self.client_model(inputs)
        to_server = self.client_intermediate.detach().requires_grad_()

        return to_server, output

    def client_backward(self, grad_from_server=None):
        if grad_from_server != None:
            self.grad_from_server = grad_from_server
            self.client_intermediate.backward(self.grad_from_server)
        else:
            self.client_intermediate.backward()

    def train(self):
        self.client_model.train()

    def eval(self):
        self.client_model.eval()


class Server(nn.Module):
    def __init__(self, server_model):
        super().__init__()
        self.server_model = server_model
        self.to_server = None
        self.grad_to_client = None

    def forward(self, to_server):
        self.to_server = to_server
        outputs = self.server_model(self.to_server)
        return outputs

    def server_backward(self,):
        self.grad_to_client = self.to_server.grad.clone()
        return self.grad_to_client

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()


class SplitNN(nn.Module):
    def __init__(
        self,
        client_list,
        server,
        client_opt_list,
        server_opt,
        scheduler_list_alice,
        scheduler_bob,
        interrupted=False,
        avg=False,
    ):
        super().__init__()
        self.clients = nn.ModuleDict({})
        self.client_opts = {}

        for i in range(len(client_list)):
            self.clients[f"alice{i}"] = Client(client_model=client_list[i])
            self.client_opts[f"alice{i}"] = client_opt_list[i]

        self.bob = nn.ModuleList([Server(server_model=server)])
        self.opt_bob = server_opt

        self.to_server = None
        self.interrupted = interrupted
        self.scheduler_list_alice = scheduler_list_alice
        self.scheduler_bob = scheduler_bob
        self.avg = avg

    def forward(self, inputs, i, out, flops):
        to_server, output = self.clients[f"alice{i}"](inputs)
        flops += (
            FlopCountAnalysis(self.clients[f"alice{i}"], inputs=inputs)
            .unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            .total()
        )
        if not self.interrupted or out:
            output_final = self.bob[0](to_server)
            flops += (
                FlopCountAnalysis(self.bob[0], inputs=to_server)
                .unsupported_ops_warnings(False).uncalled_modules_warnings(False)
                .total()
            )
            return output, output_final, flops
        else:
            return output, to_server, flops

    def get_grad(self):
        grad_to_client = self.bob[0].server_backward()
        return grad_to_client

    def backward(self, i: int, out=False, grad_to_client=None) -> None:
        if self.avg:
            assert grad_to_client != None
            self.clients[f"alice{i}"].client_backward(grad_to_client)

        elif out or not self.interrupted:
            grad_to_client = self.bob[0].server_backward()
            self.clients[f"alice{i}"].client_backward(grad_to_client)
        else:
            self.clients[f"alice{i}"].client_backward()

    def copy_params(self, last_trained: int) -> None:
        next = (last_trained + 1) % len(self.clients)
        last_trained = self.clients[f"alice{last_trained}"]
        next_model = self.clients[f"alice{next}"]
        for (name_, W_), (name, W) in zip(
            next_model.named_parameters(), last_trained.named_parameters()
        ):
            W_.data.copy_(W.data)

    def zero_grads(self, i: int) -> None:
        self.opt_bob.zero_grad()
        self.client_opts[f"alice{i}"].zero_grad()

    def train(self, i: int) -> None:
        self.bob[0].train()
        self.clients[f"alice{i}"].train()

    def eval(self, i: int) -> None:
        self.bob[0].eval()
        self.clients[f"alice{i}"].eval()

    def step(self, i: int, out=False) -> None:
        if (not self.interrupted) or out:
            self.opt_bob.step()
            self.client_opts[f"alice{i}"].step()
        else:
            self.client_opts[f"alice{i}"].step()

    def scheduler_step(self, i, acc_bob, out=True, step_bob=False):
        #         self.scheduler_list_alice[i].step(acc_bob)
        #         if out and step_bob:
        #             self.scheduler_bob.step(acc_bob)
        self.scheduler_list_alice[i].step()
        if out and step_bob:
            self.scheduler_bob.step()

    def evaluator(self, test_loader, i):
        self.eval(i)
        correct_m1, correct_m2 = 0.0, 0.0
        total = 0.0

        for images, labels in test_loader:
            if self.interrupted:
                images = images.to("cuda:0")
                labels = labels.to("cuda:0")
            else:
                images = images.to("cuda:0")
                labels = labels.to("cuda:0")

            output, output_final, f = self.forward(
                images, i, out=True, flops=0)
            _, predicted_m1 = torch.max(output_final.data, 1)
            r, predicted_m2 = torch.max(output.data, 1)

            total += labels.size(0)
            correct_m1 += (predicted_m1 == labels).sum()
            correct_m2 += (predicted_m2 == labels).sum()

        # accuracy of bob with ith alice
        accuracy_m1 = float(correct_m1) / total
        accuracy_m2 = float(correct_m2) / total  # accuracy of ith alice

        # print('Accuracy Model 1: %f %%' % (100 * accuracy_m1))
        # print('Accuracy Model 2: %f %%' % (100 * accuracy_m2))

        return accuracy_m1, accuracy_m2


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

    opt_list_alice = []
    scheduler_list_alice = []

    for i, alice in enumerate(alice_models):
        alice.train()
        opt_list_alice.append(
            #             optim.SGD(alice.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            optim.Adam(alice.parameters(), weight_decay=1e-4)
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
        model_bob,
        opt_list_alice,
        opt_bob,
        scheduler_list_alice,
        scheduler_bob,
        interrupted=interrupted,
        avg=avg,
    )

    return split_model


def experiment_ucb(
    experiment_name,
    split_nn,
    interrupted_nn,
    train_loader_list,
    test_loader,
    interrupt_range,
    epochs,
    k,
    discount_hparam,
    dataset_sizes,
    poll_clients,
    steps=None,
    num_clients=100,
):
    flops_split, flops_interrupted, steps = 0, 0, 0
    (
        flops_split_list,
        flops_interrupted_list,
        acc_split_list,
        acc_interrupted_list,
        steps_list,
        time_list,
    ) = ([], [], [], [], [], [])
    wallclock_start = time()

    bandit = UCB(num_clients, discount_hparam, dataset_sizes, k)

    criterion = nn.CrossEntropyLoss(reduction='none')
    flag = True
    t = trange(epochs, desc="", leave=True)
    device_1, device_2 = "cuda:0", "cuda:0"
    split_nn = nn.DataParallel(split_nn).to(device_1)
    interrupted_nn = nn.DataParallel(interrupted_nn).to(device_2)

    selected_ids = random.sample(list(range(num_clients)), k)
    for ep in t:  # 200
        for b in range(len(train_loader_list[0])):
            for i in range(num_clients):  # 100
            # losses = []
            # for x, y in train_loader_list[i]:
                x, y = train_loader_list[i][b]
                # zero grads
                split_nn.module.zero_grads(i)
                interrupted_nn.module.zero_grads(i)
                split_nn.module.train(i)
                interrupted_nn.module.train(i)

                # traditional split learning
                x, y = x.to(device_1), y.to(device_1)
                _, output_final, flops_split = split_nn.module(
                    x, i, True, flops_split)
                loss1 = criterion(output_final, y)
                loss1.mean().backward()
                split_nn.module.backward(i)
                split_nn.module.step(i, out=True)

                # interrupt activation flow
                if ep in range(*interrupt_range) or i not in selected_ids:
                    x, y = x.to(device_2), y.to(device_2)
                    intermediate_output, x_next, flops_interrupted = interrupted_nn.module(
                        x, i, False, flops_interrupted
                    )
                    loss2 = criterion(intermediate_output, y)
                    losses = loss2.clone().detach().cpu().numpy()
                    if poll_clients:
                        loss_mean, loss_std = torch.mean(losses), torch.std(losses) 
                    else:
                        loss_mean, loss_std = None, None
                    loss2.mean().backward()
                    interrupted_nn.module.step(i, out=False)
                else:
                    x, y = x.to(device_2), y.to(device_2)
                    _, output_final, flops_interrupted = interrupted_nn.module(
                        x, i, True, flops_interrupted
                    )
                    loss3 = criterion(output_final, y)
                    losses = loss3.clone().detach().cpu().numpy()
                    loss_mean, loss_std = torch.mean(losses), torch.std(losses) 
                    loss3.mean().backward()
                    interrupted_nn.module.backward(i, out=True)
                    interrupted_nn.module.step(i, out=True)

                steps += 1

                bandit.update_client(i, loss_mean, loss_std, i in selected_ids)

                # copy params
                split_nn.module.copy_params(i)
                interrupted_nn.module.copy_params(i)

                selected_ids = bandit.select_clients()
                bandit.end_round()

        for i in range(num_clients):
            train_loader_list[i].shuffle()
        # do eval and record after epoch
        acc_split, alice_split = split_nn.module.evaluator(test_loader, 0)
        acc_interrupted, alice_interrupted = interrupted_nn.module.evaluator(
            test_loader, 0
        )

        # trigger scheduler bob after a warmup of 10 epochs
        if ep > 10:
            split_nn.module.scheduler_step(
                num_clients-1, acc_split, step_bob=True, out=True)
            if ep in range(*interrupt_range):
                oit = False
            else:
                oit = True
                interrupted_nn.module.scheduler_step(
                    num_clients-1, acc_interrupted, out=oit, step_bob=True)

            for i in range(0, num_clients-1):
                # trigger scheduler alices
                split_nn.module.scheduler_step(
                    i, acc_split, out=True, step_bob=False)
                if ep in range(*interrupt_range):
                    oit = False
                else:
                    oit = True
                    interrupted_nn.module.scheduler_step(
                        i, acc_interrupted, out=oit, step_bob=False)

        acc_interrupted_list.append(acc_interrupted)
        acc_split_list.append(acc_split)
        steps_list.append(steps)
        time_list.append(time() - wallclock_start)
        flops_split_list.append(flops_split)
        flops_interrupted_list.append(flops_interrupted)

        t.set_description(
            f"Split: {acc_split}, Interrupted: {acc_interrupted}, Steps: {steps}", refresh=True
        )

        if ep % 50 == 0 and ep > 0:
            if not os.path.isdir(f"runs/{experiment_name}/"):
                os.makedirs(f"runs/{experiment_name}/", exist_ok=True)
            torch.save(split_nn.state_dict(),
                       f"runs/{experiment_name}/split_nn_{steps}")
            torch.save(interrupted_nn.state_dict(),
                       f"runs/{experiment_name}/interrupted_nn_{steps}")

    return (
        acc_split_list,
        acc_interrupted_list,
        flops_split_list,
        flops_interrupted_list,
        time_list,
        steps_list,
    )


def experiment_avg_grad(
    split_nn,
    interrupted_nn,
    train_loader_list,
    test_loader,
    interrupt_range,
    epochs,
    steps=None,
    num_clients=100,
):
    flops_split, flops_interrupted, steps = 0, 0, 0
    (
        flops_split_list,
        flops_interrupted_list,
        acc_split_list,
        acc_interrupted_list,
        steps_list,
        time_list,
    ) = ([], [], [], [], [], [])
    wallclock_start = time()

    criterion = nn.CrossEntropyLoss()
    patience = 20
    early_stopping_counter = 0
    flag = True

    # create iterators
    batch_iterators = []
    for i, loader in enumerate(train_loader_list):
        batch_iterators.append(iter(loader))

    while flag:
        t = trange(epochs, desc="", leave=True)
        for ep in t:
            grad_list_split = []
            grad_list_interrupted = []
            for i in range(num_clients):
                try:
                    x, y = next(batch_iterators[i])
                except StopIteration:
                    try:
                        batch_iterators[i] = iter(train_loader_list[i])
                        x, y = next(batch_iterators[i])
                    except:
                        print("Hit unexpected Exception!")
                        continue

                x, y = x.to(device), y.to(device)
                # zero grads
                split_nn.zero_grads(i)
                interrupted_nn.zero_grads(i)

                # split learning avg grad
                _, output_final, flops_split = split_nn(
                    x, i, True, flops_split)
                loss1 = criterion(output_final, y)
                loss1.backward()
                grad = split_nn.get_grad()
                grad_list_split.append(grad)

                # interrupt activation flow
                if steps in range(*interrupt_range):
                    intermediate_output, x_next, flops_interrupted = interrupted_nn(
                        x, i, False, flops_interrupted)
                    loss2 = criterion(intermediate_output, y)
                    loss2.backward()
                    #                     interrupted_nn.backward(i, out=False)
                    interrupted_nn.step(i, out=False)

                else:
                    _, output_final, flops_interrupted = interrupted_nn(
                        x, i, True, flops_interrupted)
                    loss2 = criterion(output_final, y)
                    loss2.backward()
                    grad = interrupted_nn.get_grad()
                    grad_list_interrupted.append(grad)

                steps += 1

            # take step now
            grad_split = sum(grad_list_split) / len(grad_list_split)
            split_nn.backward(i, grad_to_client=grad_split)
            split_nn.step(0)
            del grad_list_split

            if len(grad_list_interrupted) > 0:
                grad_interrupted = sum(grad_list_interrupted) / len(
                    grad_list_interrupted
                )
                interrupted_nn.backward(
                    i, grad_to_client=grad_interrupted, out=True)
                interrupted_nn.step(0)
                del grad_list_interrupted

            if ep % 10 == 0 and ep > 0:
                # do eval and record
                for i in range(num_clients):
                    acc_split, alice_split = split_nn.evaluator(test_loader, i)
                    acc_interrupted, alice_interrupted = interrupted_nn.evaluator(
                        test_loader, i
                    )
                    acc_interrupted_list.append(acc_interrupted)
                    acc_split_list.append(acc_split)
                    steps_list.append(steps)
                    time_list.append(time() - wallclock_start)
                    flops_split_list.append(flops_split)
                    flops_interrupted_list.append(flops_interrupted)

                    # early stopping bs
                    if acc_split_list[-(early_stopping_counter + 1)] > acc_split:
                        early_stopping_counter += 1

                    # trigger scheduler
                    split_nn.scheduler_step(alice_split, acc_split, i)
                    interrupted_nn.scheduler_step(
                        alice_interrupted, acc_interrupted, i)

                    # set description
                    t.set_description(
                        f"Split: {acc_split}, Interrupted: {acc_interrupted}", refresh=True)

        # check early_stopping
        if steps > 190000:
            flag = False

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


if __name__ == "__main__":
    cifar           = True
    num_clients     = 10
    k               = 2
    discount        = 0.7
    poll_clients    = False
    ds              = "cifar10" if cifar else "tiny_imagenet"
    experiment_name = f"{ds}_ucb_k_{k}_num_clients_{num_clients}_discount_{discount}"
    interrupted     = True # Local Parallelism OFF

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

    cifar_test_loader_list = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
    )
    batch_size = 256

    # split dataset into num_clients
    cifar_train_loader_list = []
    train_sizes = torch.zeros((num_clients,))
    for i in range(num_clients):
        torch.manual_seed(i)
        train_size = int(len(trainset) / num_clients)
        buffer = len(trainset) - train_size
        ts, test_split = torch.utils.data.random_split(
            trainset, [train_size, buffer])
        cifar_train_loader = DataWrapper(
            ts,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )
        cifar_train_loader.shuffle()
        cifar_train_loader_list.append(cifar_train_loader)
        train_sizes[i] = train_size

    epochs = 150
    if interrupted:
        interrupt_range = [0, int(0.75*epochs)]
    else:
        interrupt_range = [-2, 0]  # Hack for not using Local Parallelism

    split_nn = get_model(num_clients=num_clients,
                         interrupted=False, cifar=cifar)
    interrupted_nn = get_model(
        num_clients=num_clients, interrupted=interrupted, cifar=cifar)

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
        interrupted_nn,
        cifar_train_loader_list,
        cifar_test_loader_list,
        k=3,
        poll_clients=poll_clients,
        discount_hparam=0.7,
        dataset_sizes=train_sizes,
        interrupt_range=interrupt_range,
        epochs=epochs,
        num_clients=num_clients,
    )

    import json

    out_dict = {
        "SplitNN Accuracy": acc_split_list,
        "Interrupted Accuracy": acc_interrupted_list,
        "Flops SplitNN": flops_split_list,
        "Flops Interrupted": flops_interrupted_list,
        "Time": time_list,
        "Steps": steps_list
    }
    print(out_dict)

    out_dict = {
        "SplitNN Accuracy": str(acc_split_list),
        "Interrupted Accuracy": str(acc_interrupted_list),
        "Flops SplitNN": str(flops_split_list),
        "Flops Interrupted": str(flops_interrupted_list),
        "Time": str(time_list),
        "Steps": steps_list,
    }

    with open(f"stats/{ds}/{experiment_name}.json", "w") as f:
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
