from functools import partial
from operator import sub
import os
import pdb
import glob
import numpy as np
from misc.utils import *
from dataclasses import dataclass
import torch
import math


@dataclass
class Arguments:
    base_dir: str
    num_clients: int


class DataLoader:
    """ Data Loader

    Loading data for the corresponding clients

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)

    Modified by: 
        Surya Kant Sahu (surya.oju@pm.me)
    """

    def __init__(self, args):
        self.args = args
        self.base_dir = args.base_dir
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }

        files = os.listdir(self.args.base_dir)
        files = ["_".join(f.replace(".npy", "").split("_")[:-1])
                 for f in files]
        files = sorted(list(set(files)))
        idx = [28, 30, 38, 45, 10, 20, 48, 23, 32, 33, 19, 18, 29, 42, 22, 16, 49, 8, 14, 24, 9, 26, 35, 31, 3, 40, 1, 21, 13, 4, 44, 6, 2, 43, 25, 37, 12, 27, 0, 11, 46, 36, 41, 7, 34, 17, 15, 39, 5, 47]
        files = [files[i] for i in idx]
        self.task_set = {
            k: [v] for k, v in enumerate(files[:self.args.num_clients])
        }

    def get_train(self, cid, task_id):
        return load_task(self.base_dir, self.task_set[cid][task_id]+'_train.npy').item()

    def get_valid(self, cid, task_id):
        valid = load_task(
            self.base_dir, self.task_set[cid][task_id]+'_valid.npy').item()
        return valid['x_valid'], valid['y_valid']

    def get_test(self, cid, task_id):
        x_test_list = []
        y_test_list = []
        for tid, task in enumerate(self.task_set[cid]):
            if tid <= task_id:
                test = load_task(self.base_dir, task+'_test.npy').item()
                x_test_list.append(test['x_test'])
                y_test_list.append(test['y_test'])
        return x_test_list, y_test_list


class ShuffledCycle:
    def __init__(self, indices):
        self.indices = indices
        self.i = 0
        random_shuffle(77, self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.indices):
            self.i = 0
            random_shuffle(77, self.indices)
        self.i += 1
        return self.indices[self.i-1]


class NonIID50Train:
    def __init__(self, num_clients, client_id, batch_size):

        self.num_clients = num_clients
        self.client_id = client_id
        self.batch_size = batch_size

        args = Arguments("./non_iid_50/scripts/tasks/non_iid_50/", num_clients)
        self.dl = dl = DataLoader(args)
        self.data = {}
        for k in range(num_clients):
            d = dl.get_train(k, 0)
            self.data[k] = {
                "x": d['x_train'],
                "y": d['y_train']
            }

        self.ds_sizes = {k: v["x"].shape[0] for k, v in self.data.items()}
        self.index_cycles = {k: ShuffledCycle(
            list(range(v))) for k, v in self.ds_sizes.items()}

    def __len__(self):
        return int(math.ceil(max(self.ds_sizes.values())//self.batch_size))

    def get_one_example(self):
        c = self.client_id
        idx = next(self.index_cycles[c])
        xc = self.data[c]["x"][idx].transpose(2, 0, 1)
        yc = one_hot_to_int(self.data[c]["y"][idx])
        return xc, yc

    def __getitem__(self, *args):
        items = []
        for _ in range(self.batch_size):
            items.append(self.get_one_example())
        return non_iid_50_collate_fn(items)

    def shuffle(self):
        pass


class NonIID50Test(torch.utils.data.Dataset):
    def __init__(self, num_clients, client_id):

        self.num_clients = num_clients
        self.client_id = client_id

        args = Arguments("./non_iid_50/scripts/tasks/non_iid_50/", num_clients)
        self.dl = dl = DataLoader(args)
        self.data = {}
        for k in range(num_clients):
            x, y = dl.get_test(k, 0)
            self.data[k] = {
                "x": x[0],
                "y": y[0]
            }

        self.ds_sizes = {k: v["x"].shape[0] for k, v in self.data.items()}
        self.index_cycles = {k: ShuffledCycle(
            list(range(v))) for k, v in self.ds_sizes.items()}

    def __len__(self):
        return self.ds_sizes[self.client_id]

    def __getitem__(self, *args):
        c = self.client_id
        idx = next(self.index_cycles[c])
        xc = self.data[c]["x"][idx].transpose(2, 0, 1)
        yc = one_hot_to_int(self.data[c]["y"][idx])

        return xc, yc

    def shuffle(self):
        pass


def one_hot_to_int(onehot):
    for i in range(len(onehot)):
        if onehot[i] > 0.:
            return i


def non_iid_50_collate_fn(batches):
    xb, yb = [], []
    for x, y in batches:
        xb.append(x)
        yb.append(y)
    xb = torch.FloatTensor(np.stack(xb)).contiguous()
    yb = torch.LongTensor(np.stack(yb)).contiguous()
    return xb, yb


def get_non_iid_50(batch_size, num_workers, num_clients):
    tr_dl = {}
    for i in range(num_clients):
        tr_ds = NonIID50Train(num_clients, i, batch_size)
        tr_dl[i] = tr_ds

    ts_dl = {}
    for i in range(num_clients):
        ts_ds = NonIID50Test(num_clients, client_id=i)
        ts_dl[i] = torch.utils.data.DataLoader(
            ts_ds, 256, num_workers=num_workers, collate_fn=non_iid_50_collate_fn, pin_memory=True)
    return tr_dl, ts_dl


if __name__ == "__main__":
    bdir = "./non_iid_50/scripts/tasks/non_iid_50/"
    # args = Arguments(
    #     bdir,
    #     10
    #     )
    # dl = DataLoader(args)
    # print("train")
    # for i in range(10):
    #     x = dl.get_train(i, 0)
    #     print(x['x_train'].shape)
    # print("test")
    # for i in range(10):
    #     x, y = dl.get_test(i, 0)
    #     print(x[0].shape)

    # tr_ds = NonIID50Train(10, 2, 32)
    # for i in range(len(tr_ds)):
    #     x, y = tr_ds[i]
    #     print(x.shape, y.shape)

    # ts_ds = NonIID50Test(10, client_id=1)
    # ts_dl = torch.utils.data.DataLoader(ts_ds, 32, num_workers=4, collate_fn=non_iid_50_collate_fn)
    # for x, y in ts_dl:
    #     print(x.shape, y.shape)

    tr_dl, ts_dl = get_non_iid_50(32, 4, 10)
    # for x, y in ts_dl[0]:
    #     print(x.shape, y.shape)

    for i in range(len(tr_dl[0])):
        x, y = tr_dl[0][i]
        # print(out)
        print(x.shape, y.shape)
