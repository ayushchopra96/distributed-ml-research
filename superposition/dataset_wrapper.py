from functools import lru_cache
import torch
from triplettorch import TripletDataset

import numpy as np
from torchvision import datasets, transforms


def noniid(dataset_num_classes, train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(
        len(train_dataset)/num_samples), num_samples
    num_classes = dataset_num_classes
    num_imgs_perc_test, num_imgs_test_total = 1000, len(test_dataset)
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        # print(user_labels_set)
        # print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(
                label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)
        # print(set(labels_test_raw[dict_users_test[i].astype(int)]))

    return dict_users_train, dict_users_test


class ClientNonIID(torch.utils.data.Dataset):
    def __init__(self, ds, user_to_idx, train):
        self.ds = ds
        self.user_to_idx = user_to_idx
        self.num_clients = len(self.user_to_idx)
        self.train = train
        self.num_samples_cl = 4
        self.prep_datasets()

    def prep_datasets(self):
        partitioned_tr = []
        cl_ds = []
        train_sizes = torch.zeros((self.num_clients,))

        for i in range(self.num_clients):
            train_idx = self.user_to_idx[i].astype(int)
            train_size = len(train_idx)

            if self.train:

                labels = [self.ds[k][1] for k in train_idx]
                def fn(index): return self.ds[train_idx[index]][0].unsqueeze(
                    0).numpy()
                contrastive = TripletDataset(
                    labels, fn, len(train_idx), self.num_samples_cl)
                cl_ds.append(contrastive)

            partitioned_tr.append(train_idx)
            train_sizes[i] = train_size

        self.cl_ds = cl_ds
        self.partitioned = partitioned_tr
        self.train_sizes = train_sizes

    def __len__(self):
        return max([len(self.partitioned[i]) for i in range(self.num_clients)])

    def __getitem__(self, idx):
        if self.train:
            return self.get_item_clf_cl(idx)
        else:
            return self.get_item_clf(idx)

    @lru_cache()
    def get_dummy(self, ds):
        if ds == "clf":
            return self.ds[self.partitioned[0][0]]
        else:
            return self.cl_ds[0][0]

    def clf_item_getter(self, client_id, idx):
        def f():
            return self.ds[self.partitioned[client_id][idx]]
        return f

    def cl_item_getter(self, client_id, idx):
        def f():
            return self.cl_ds[client_id][idx]
        return f

    def safe_get_example(self, ds, client_id, idx):
        if ds == "clf":
            item_getter = self.clf_item_getter(client_id, idx)
            dummy_getter = self.get_dummy("clf")
        else:
            item_getter = self.cl_item_getter(client_id, idx)
            dummy_getter = self.get_dummy("cl")

        try:
            x, y = item_getter()
            mask = 0.
        except Exception as e:
            print(e)
            x_dummy, y_dummy = dummy_getter()
            x, y = torch.zeros_like(x_dummy), torch.zeros_like(y_dummy)
            mask = 1.
        return x, y, mask

    def get_item_clf(self, idx):
        out_x = []
        out_y = []
        client_mask = torch.zeros((self.num_clients,))
        for i in range(self.num_clients):
            x, y, mask = self.safe_get_example("clf", i, idx)
            out_x.append(x.unsqueeze(0))
            out_y.append(torch.tensor(y).unsqueeze(0))

            client_mask[i] = mask
        out_x = torch.cat(out_x, dim=0)
        out_y = torch.cat(out_y, dim=0)
        client_mask = ~(client_mask.bool())
        return out_x, out_y, client_mask

    def get_item_clf_cl(self, idx):
        out_x = []
        out_y = []
        out_xc = []
        out_yc = []
        client_mask = torch.zeros((self.num_clients,))
        for i in range(self.num_clients):
            x, y, mask = self.safe_get_example("clf", i, idx)
            out_x.append(x.unsqueeze(0))
            out_y.append(torch.tensor(y).unsqueeze(0))

            yc, xc, mask = self.safe_get_example("cl", i, idx)
            out_xc.append(xc)
            out_yc.append(yc.unsqueeze(1))

            client_mask[i] = mask

        out_x = torch.cat(out_x, dim=0)
        out_y = torch.cat(out_y, dim=0)
        out_xc = torch.cat(out_xc, dim=1)
        out_yc = torch.cat(out_yc, dim=1)
        client_mask = ~(client_mask.bool())
        return out_x, out_y, out_xc, out_yc, client_mask


class ClassificationContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, num_clients, ds, use_contrastive, train):
        self.num_clients = num_clients
        self.use_contrastive = use_contrastive
        self.train = train

        self.prep_datasets(ds)

    def prep_datasets(self, ds):
        partitioned_tr = []
        partitioned_ts = []
        cl_ds = []
        train_sizes = torch.zeros((self.num_clients,))

        for i in range(self.num_clients):
            torch.manual_seed(i)
            train_size = int(len(ds) / self.num_clients)
            buffer = len(ds) - train_size
            train, test_split = torch.utils.data.random_split(
                ds, [train_size, buffer])

            labels = [train[i][1] for i in range(len(train))]
            def fn(index): return train[index][0].unsqueeze(0).numpy()
            contrastive = TripletDataset(labels, fn, len(train), 4)

            cl_ds.append(contrastive)
            partitioned_tr.append(train)
            partitioned_ts.append(test_split)

            train_sizes[i] = train_size

        self.cl_ds = cl_ds
        self.partitioned_tr = partitioned_tr
        self.partitioned_ts = partitioned_ts
        self.train_sizes = train_sizes

    def __len__(self):
        if self.train:
            return len(self.partitioned_tr[0])
        else:
            return len(self.partitioned_ts[0])

    def __getitem__(self, idx):
        if self.train:
            ds = self.partitioned_tr
        else:
            ds = self.partitioned_ts

        out_x = []
        out_y = []
        out_xc = []
        out_yc = []
        for i in range(self.num_clients):
            x, y = ds[i][idx]
            out_x.append(x.unsqueeze(0))
            out_y.append(torch.tensor(y).unsqueeze(0))

            yc, xc = self.cl_ds[i][idx]
            out_xc.append(xc)
            out_yc.append(yc.unsqueeze(1))

        out_x = torch.cat(out_x, dim=0)
        out_y = torch.cat(out_y, dim=0)
        out_xc = torch.cat(out_xc, dim=1)
        out_yc = torch.cat(out_yc, dim=1)
        return out_x, out_y, out_xc, out_yc, None


def collate_fn_v2_test(batches):
    xb, yb, maskb = [], [], []

    for x, y, mask in batches:

        xb.append(x.unsqueeze(0))
        yb.append(y.unsqueeze(0))

        maskb.append(mask.unsqueeze(0))

    xb = torch.cat(xb, dim=0)
    yb = torch.cat(yb, dim=0)

    maskb = torch.cat(maskb, dim=0)
    return xb, yb, maskb


def collate_fn_v2_train(batches):
    xb, yb, xcb, ycb, maskb = [], [], [], [], []

    for x, y, xc, yc, mask in batches:

        xb.append(x.unsqueeze(0))
        yb.append(y.unsqueeze(0))

        xcb.append(xc)
        ycb.append(yc)

        maskb.append(mask.unsqueeze(0))

    xb = torch.cat(xb, dim=0)
    yb = torch.cat(yb, dim=0)

    xcb = torch.cat(xcb, dim=0)
    ycb = torch.cat(ycb, dim=0)

    maskb = torch.cat(maskb, dim=0)
    return xb, yb, xcb, ycb, maskb


def collate_fn_v1(batches):
    xb, yb, xcb, ycb = [], [], [], [], []

    for x, y, xc, yc, mask in batches:

        xb.append(x.unsqueeze(0))
        yb.append(y.unsqueeze(0))

        xcb.append(xc)
        ycb.append(yc)

    xb = torch.cat(xb, dim=0)
    yb = torch.cat(yb, dim=0)

    xcb = torch.cat(xcb, dim=0)
    ycb = torch.cat(ycb, dim=0)

    return xb, yb, xcb, ycb, None
