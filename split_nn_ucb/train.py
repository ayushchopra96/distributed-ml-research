from argparse import ArgumentParser
from dataclasses import dataclass

from fvcore.nn.flop_count import flop_count
from models_cifar import resnet32, test
from ucb import UniformRandom, BayesianUCB, VWBandit
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import h5py
from time import time
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
from dataset_wrapper import DataWrapper, DataWrapper, classwise_subset
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PartitionedResNets import resnet32 as partitioned_resnet32
from PartitionedResNets import PartitionedResNet
from MaskedResNets import resnet32 as masked_resnet32
from MaskedResNets import MaskedResNet
from functools import lru_cache, partial
from non_iid_50 import get_non_iid_50

from utils import *
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.enabled = False
torch.manual_seed(123)
# torch.autograd.set_detect_anomaly(True)

# from tqdm.notebook import


# from tqdm.notebook import tqdm

# from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def forward(self, to_server, i=None):
        self.to_server = to_server
        if i is not None:
            outputs = self.server_model(self.to_server, i)
        else:
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
        self.is_partitioned_or_masked = isinstance(
            server, PartitionedResNet) or isinstance(server, MaskedResNet)
        self.to_server = None
        self.interrupted = interrupted
        self.scheduler_list_alice = scheduler_list_alice
        self.scheduler_bob = scheduler_bob
        self.avg = avg

    def forward(self, inputs, i, out, flops):
        to_server, output = self.clients[f"alice{i}"](inputs)
        client_flops = compute_flops(self.clients[f"alice{i}"], (inputs,), "alice")
        if out:
            if self.is_partitioned_or_masked:
                output_final = self.bob[0](to_server, i)
                flops += compute_flops(self.bob[0], (to_server, i), "bob")
                return output, output_final, flops + client_flops, client_flops
            else:
                output_final = self.bob[0](to_server)
                flops += compute_flops(self.bob[0], (to_server,), "bob")
                return output, output_final, flops + client_flops, client_flops
        else:
            return output, to_server, flops + client_flops

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
            return grad_to_client
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

    @torch.no_grad()
    def evaluator(self, test_loader, i):
        self.eval(i)
        correct_m1, correct_m2 = 0.0, 0.0
        total = 0.0

        for images, labels in test_loader:
            if self.interrupted:
                images = images.to(device)
                labels = labels.to(device)
            else:
                images = images.to(device)
                labels = labels.to(device)

            output, output_final, f, fc = self.forward(
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


def get_model(num_clients=100, num_partitions=1, use_masked=False, interrupted=False, avg=False, cifar=True, emb_dim=None, non_iid_50=False):
    assert(not (num_partitions > 1 and use_masked))
    if cifar:
        num_classes = 10
        option = 'A'
        T_max = 160
    elif non_iid_50:
        num_classes = 5
        option = 'A'
        T_max = 160
    
    alice_models = []
    client_model_fn = partial(resnet32, hooked=True, num_classes=num_classes, emb_dim=emb_dim)
    for i in range(num_clients):
        model_a = client_model_fn()
        alice_models.append(model_a)

    opt_list_alice = []
    scheduler_list_alice = []

    for i, alice in enumerate(alice_models):
        alice.train()
        opt_list_alice.append(
            #             optim.SGD(alice.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            optim.Adam(alice.parameters(), lr=1e-3, weight_decay=1e-4)
        )
        scheduler_list_alice.append(
            CosineAnnealingLR(opt_list_alice[-1], T_max=T_max)
            #             ReduceLROnPlateau(opt_list_alice[-1], mode='max', factor=0.7, patience=5)
        )
    if num_partitions > 1:
        print("Using Partitioned")
        model_bob = partitioned_resnet32(
            num_partitions=num_partitions, num_clients=num_clients, hooked=False, num_classes=num_classes)
    elif use_masked:
        print("Using Masked")
        model_bob = masked_resnet32(
            num_masks=num_clients, num_clients=num_clients, hooked=False, num_classes=num_classes)
    else:
        model_bob = resnet32(hooked=False, num_classes=num_classes)
    opt_bob = optim.Adam(model_bob.parameters(), lr=1e-3, weight_decay=1e-4)
    # shared = []
    # client_specific = []
    # for pname, p in model_bob.named_parameters():
    #     if pname.find("masks") != -1:
    #         client_specific.append(p)
    #     else:
    #         shared.append(p)
    # opt_bob = optim.Adam(
    #     [{'params': shared, 'lr': 3e-3}, 
    #     {'params': client_specific}], lr=3e-3)
    scheduler_bob = CosineAnnealingLR(opt_bob, T_max=T_max)
    # scheduler_bob = ReduceLROnPlateau(opt_bob, mode='max', factor=0.7, patience=5)

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

    return split_model, client_model_fn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_interrupt_or_not(use_ucb, selected_ids, id, interrupt_range, ep):
    ucb_interrupt = False

    in_interrupt_range = False
    if ep in range(*interrupt_range):
        in_interrupt_range = True

    if id not in selected_ids:
        ucb_interrupt = True

    return (ucb_interrupt and use_ucb) or in_interrupt_range

from pytorch_metric_learning import miners as pml_miners, losses as pml_losses

def experiment_ucb(
    experiment_name,
    alpha,
    vanilla,
    split_nn,
    interrupted_nn,
    train_loader_list,
    contrastive_list,
    test_loader,
    avg_clients,
    model_fn,
    interrupt_range,
    epochs,
    k,
    use_ucb,
    use_random,
    use_vw,
    use_contrastive,
    discount_hparam,
    dataset_sizes,
    poll_clients,
    l1_norm_weight,
    steps=None,
    num_clients=100,
):

    contrastive_loss = pml_losses.SupConLoss()

    flops_split_client, flops_split, flops_interrupted, comm_split, comm_interrupted, steps = 0, 0, 0, 0, 0, 0
    (   
        flops_split_client_list,
        flops_split_list,
        flops_interrupted_list,
        acc_split_list,
        acc_interrupted_list,
        comm_split_list,
        comm_interrupted_list,
        steps_list,
        time_list,
    ) = ([], [], [], [], [], [], [], [], [])

    wallclock_start = time()

    final_ce_meter = AverageMeter()
    intermediate_ce_meter = AverageMeter()
    cl_meter = AverageMeter()

    if use_random:
        bandit = UniformRandom(num_clients, discount_hparam, dataset_sizes, k)
    elif use_ucb:
        bandit = BayesianUCB(num_clients, discount_hparam, dataset_sizes, k)
    elif use_vw:
        bandit = VWBandit(num_clients, discount_hparam, dataset_sizes, k)
    else:
        bandit = UniformRandom(num_clients, discount_hparam, dataset_sizes, num_clients)
    if vanilla:
        alpha = 0.
    criterion = nn.CrossEntropyLoss(reduction='none')
    flag = True
    t = trange(epochs, desc="", leave=True)
    device_1, device_2 = device, device
    # split_nn = nn.DataParallel(split_nn, output_device=device_1)
    # split_nn.module.cuda()

    model_averager = ModelAverager(model_fn)
    interrupted_nn = nn.DataParallel(interrupted_nn, output_device=device_2)
    interrupted_nn.module.cuda()
    if isinstance(train_loader_list, dict):
        num_batches = max([len(tr_dl) for tr_dl in train_loader_list.values()])
    else:
        num_batches = max([len(tr_dl) for tr_dl in train_loader_list])
    
    selected_ids = random.sample(list(range(num_clients)), k)
    for ep in t:  # 200
        batch_iter = trange(num_batches)
        for b in batch_iter:
            for i in range(num_clients):  # 100
                total_loss = 0.

                x, y = train_loader_list[i][b]
                # zero grads
                # split_nn.module.zero_grads(i)
                interrupted_nn.module.zero_grads(i)
                # split_nn.module.train(i)
                interrupted_nn.module.train(i)

                if not vanilla and to_interrupt_or_not(use_ucb or use_random, selected_ids, i, interrupt_range, ep):
                    # interrupt activation flow
                    if use_contrastive:
                        x, y = contrastive_list[i][b]
                        b, num_views = x.shape[0], x.shape[1]
                    x, y = x.to(device_2), y.to(device_2)
                    before_flops = deepcopy(flops_interrupted)
                    intermediate_output, x_next, flops_interrupted = interrupted_nn.module(
                        x, i, False, flops_interrupted
                    )
                    flops_split_client += (flops_interrupted - before_flops)
                    if use_contrastive:
                        emb = intermediate_output.reshape(b, -1)
                        # emb = x_next.reshape(b, -1)
                        loss2 = contrastive_loss(emb, y)
                        cl_meter.update(loss2.cpu().detach().mean().item())
                    else:
                        loss2 = criterion(x_next, y)
                        intermediate_ce_meter.update(loss2.mean().item())
                    losses = loss2.clone().detach().cpu()
                    if poll_clients:
                        loss_mean, loss_std = torch.mean(
                            losses).item(), torch.std(losses).item()
                        comm_interrupted += 2 * 4 * 1e-6
                    else:
                        loss_mean, loss_std = None, None
                    total_loss = total_loss + loss2.mean()
                    total_loss.backward()
                    interrupted_nn.module.step(i, out=False)

                else:
                    x, y = x.to(device_2), y.to(device_2)
                    comm_interrupted += compute_comm_cost(x)
                    _, output_final, flops_interrupted, flops_client = interrupted_nn.module(
                        x, i, True, flops_interrupted
                    )
                    flops_split_client += flops_client
                    loss3 = criterion(output_final, y)
                    losses = loss3.clone().detach().cpu()
                    loss_mean, loss_std = torch.mean(
                        losses).item(), torch.std(losses).item()
                    total_loss = total_loss + loss3.mean()
                    final_ce_meter.update(loss3.mean().item())
                    if isinstance(interrupted_nn.module.bob[0].server_model, MaskedResNet):
                        sparsity = interrupted_nn.module.bob[0].server_model.sparsity_constraint(
                        )
                        total_loss = total_loss + l1_norm_weight * sparsity
                    total_loss.backward()
                    grad = interrupted_nn.module.backward(i, out=True)
                    interrupted_nn.module.step(i, out=True)
                    comm_interrupted += compute_comm_cost(grad)

                steps += 1
                if ep not in range(*interrupt_range):
                    bandit.update_client(i, loss_mean, loss_std, i in selected_ids)

                # copy params
                interrupted_nn.module.copy_params(i)

            if ep not in range(*interrupt_range):
                bandit.end_round()
            selected_ids = bandit.select_clients()

            if use_contrastive:
                batch_iter.set_description(
                    f"Final CE: {final_ce_meter.avg:.2f}, CL: {cl_meter.avg:.2f}", refresh=True
                )
            else:
                batch_iter.set_description(
                    f"Final CE: {final_ce_meter.avg:.2f}, Intermediate CE: {intermediate_ce_meter.avg:.2f}", refresh=True
                )

        final_ce_meter.reset()
        cl_meter.reset()
        intermediate_ce_meter.reset()


        for i in range(num_clients):
            train_loader_list[i].shuffle()
            if use_contrastive:
                contrastive_list[i].shuffle()

        # do eval and record after epoch
        # acc_split, alice_split = split_nn.module.evaluator(test_loader, 0)
        accs_final, accs_alice = [], []
        for i in range(num_clients):
            if isinstance(test_loader, list) or isinstance(test_loader, dict):
                loader = test_loader[i]
            else:
                loader = test_loader
            acc_interrupted, alice_interrupted = interrupted_nn.module.evaluator(
                loader, i
            )
            accs_final.append(acc_interrupted)
            accs_alice.append(alice_interrupted)
        accs_final = np.mean(accs_final)
        accs_alice = np.mean(accs_alice)

        # trigger scheduler bob after a warmup of 10 epochs
        if ep > 10:
            # split_nn.module.scheduler_step(
            #     num_clients-1, acc_split, step_bob=True, out=True)
            if ep in range(*interrupt_range):
                oit = False
            else:
                oit = True
                interrupted_nn.module.scheduler_step(
                    num_clients-1, accs_final, out=oit, step_bob=True)

            for i in range(0, num_clients-1):
                # trigger scheduler alices
                # split_nn.module.scheduler_step(
                #     i, acc_split, out=True, step_bob=False)
                if ep in range(*interrupt_range):
                    oit = False
                else:
                    oit = True
                    interrupted_nn.module.scheduler_step(
                        i, accs_final, out=oit, step_bob=False)

            if avg_clients:
                comm_interrupted += model_averager.average(interrupted_nn.module)


        acc_interrupted_list.append(accs_final)
        acc_split_list.append(0.)
        steps_list.append(steps)
        time_list.append(time() - wallclock_start)
        flops_split_list.append(flops_split)
        flops_split_client_list.append(flops_split_client);
        flops_interrupted_list.append(flops_interrupted)
        comm_split_list.append(comm_split)
        comm_interrupted_list.append(comm_interrupted)

        t.set_description(
            f"Method: {accs_final}, Steps: {steps}", refresh=True
        )

        # if ep % 50 == 0 and ep > 0:
        #     if not os.path.isdir(f"runs/{experiment_name}/"):
        #         os.makedirs(f"runs/{experiment_name}/", exist_ok=True)
        #     torch.save(split_nn.state_dict(),
        #                f"runs/{experiment_name}/split_nn_{steps}")
        #     torch.save(interrupted_nn.state_dict(),
        #                f"runs/{experiment_name}/interrupted_nn_{steps}")

    return (
        acc_split_list,
        acc_interrupted_list,
        flops_split_list,
        flops_interrupted_list,
        flops_split_client_list,
        time_list,
        steps_list,
        comm_split_list,
        comm_interrupted_list,
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
    non_iid_50: bool = True
    num_clients: int = 10
    k: int = 3
    discount: float = 0.85
    poll_clients: bool = False
    interrupted: bool = True  # Interruption OFF/ON
    batch_size: int = 32
    epochs: int = 100
    use_ucb: bool = True
    use_random: bool = True
    use_vw: bool = True
    use_contrastive: bool = True
    num_partitions: int = 1
    use_masked: bool = False
    l1_norm_weight: float = 1e-6
    classwise_subset: bool = False
    num_groups: int = 5
    experiment_name: str = ""
    interrupt_range: float = 0.75
    emb_dim: int = 64
    alpha: float = 0.3
    vanilla: bool = True
    avg_clients: bool = True

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

    if hparams_.experiment_name == "":
        experiment_name = "-".join(temp)
    else:
        experiment_name = hparams_.experiment_name + str(hparams_.num_clients)
    print(experiment_name)

    niid_classes = None
    
    cifar = hparams_.cifar
    num_clients = hparams_.num_clients
    k = hparams_.k
    discount = hparams_.discount
    poll_clients = hparams_.poll_clients
    interrupted = hparams_.interrupted  # Interruption OFF/ON
    batch_size = hparams_.batch_size
    epochs = hparams_.epochs
    interrupt_range = hparams_.interrupt_range
    emb_dim = hparams_.emb_dim

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

    if hparams_.non_iid_50:
        cifar_train_loader_list, cifar_test_loader_list = get_non_iid_50(
            batch_size, 8, hparams_.num_clients)
        contrastive_dataset_list = [None] * hparams_.num_clients
        if hparams_.use_contrastive:
            contrastive_dataset_list = cifar_train_loader_list

        train_sizes = np.array(
            list(cifar_train_loader_list[0].ds_sizes.values()))
        niid_classes = 5

    elif hparams_.classwise_subset:
        # trainset = torchvision.datasets.CIFAR100(
        #     root="./data", train=True, download=True, transform=transform
        # )
        # testset = torchvision.datasets.CIFAR100(
        #     root="./data", train=False, download=True, transform=transform_val
        # )
        total = torch.utils.data.ConcatDataset([trainset, testset])
        train_idx, test_idx, train_sizes = classwise_subset(
            total,
            hparams_.num_clients,
            hparams_.num_groups,
            0.1
        )
        niid_classes = 2
        cifar_test_loader_list = []
        cifar_train_loader_list = []
        contrastive_dataset_list = []
        for c in range(hparams_.num_clients):
            ts = torch.utils.data.Subset(total, train_idx[c])
            train_dl = DataWrapper(
                ts,
                batch_size=batch_size,
                num_workers=os.cpu_count(),
                pin_memory=not cifar,
            )
            train_dl.shuffle()
            cifar_train_loader_list.append(train_dl)

            if hparams_.use_contrastive:
                contrastive_dataset_list.append(train_dl)
            else:
                contrastive_dataset_list.append(None)

            test_dl = torch.utils.data.DataLoader(
                total, batch_size=256, num_workers=os.cpu_count(), pin_memory=not cifar,
                sampler=torch.utils.data.SubsetRandomSampler(test_idx[c])
            )
            cifar_test_loader_list.append(test_dl)
    else:
        cifar_test_loader_list = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=os.cpu_count(), pin_memory=not cifar
        )

        # split dataset into num_clients
        cifar_train_loader_list = []
        contrastive_dataset_list = []
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
                pin_memory=not cifar,
            )
            cifar_train_loader.shuffle()
            cifar_train_loader_list.append(cifar_train_loader)
            if hparams_.use_contrastive:
                contrastive_dataset_list.append(cifar_train_loader)
            else:
                contrastive_dataset_list.append(None)

            train_sizes[i] = train_size

    if interrupted:
        interrupt_range = [0, int(interrupt_range*epochs)]
    else:
        interrupt_range = [-2, 0]  # Hack for not using Local Parallelism
        emb_dim = None
    # split_nn = get_model(num_clients=num_clients,
    #                      interrupted=False, cifar=cifar)
    interrupted_nn, model_fn = get_model(num_clients=num_clients, num_partitions=hparams_.num_partitions, use_masked=hparams_.use_masked, interrupted=interrupted, cifar=cifar, emb_dim=emb_dim, non_iid_50=hparams_.non_iid_50)

    print("Average Train set size per client: ", train_sizes.mean().item())
    (
        acc_split_list,
        acc_interrupted_list,
        flops_split_list,
        flops_interrupted_list,
        flops_split_client_list,
        time_list,
        steps_list,
        comm_split_list,
        comm_interrupted_list,
    ) = experiment_ucb(
        experiment_name,
        hparams_.alpha,
        hparams_.vanilla,
        None,
        interrupted_nn,
        cifar_train_loader_list,
        contrastive_dataset_list,
        cifar_test_loader_list,
        avg_clients=hparams_.avg_clients,
        model_fn=model_fn,
        k=hparams_.k,
        use_contrastive=hparams_.use_contrastive,
        use_ucb=hparams_.use_ucb,
        use_random=hparams_.use_random,
        use_vw=hparams_.use_vw,
        poll_clients=poll_clients,
        discount_hparam=hparams_.discount,
        dataset_sizes=train_sizes,
        interrupt_range=interrupt_range,
        l1_norm_weight=hparams_.l1_norm_weight,
        epochs=epochs,
        num_clients=num_clients,
    )

    import json

    out_dict = {
        "Method Accuracy": str(acc_interrupted_list),
        "Method Flops": str(flops_interrupted_list),
        "Method Client Flops": str(flops_split_client_list),
        "Method Comm Cost": str(comm_interrupted_list),
        "Time": str(time_list),
        "Steps": str(steps_list),
        "hparams": hparams_.__dict__
    }
    print(out_dict)

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
