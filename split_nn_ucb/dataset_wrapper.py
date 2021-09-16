import torch
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split

class DataWrapper:
    def __init__(self, dataset, batch_size, *args, **kwargs):
        self.ds = dataset
        self.indices = list(range(len(self.ds)))
        self.num_batches = math.ceil(len(self.ds) / batch_size)
        self.batch_size = batch_size

        self.complete_batches = len(self.ds) // batch_size
        self.incomplete_batches = len(self.ds) % batch_size

    def shuffle(self):
        idx = list(range(len(self.ds)))
        random.shuffle(idx)
        splitted = np.array_split(idx, self.num_batches)
        incomplete_batch = []
        complete_batches = []
        for b in range(0, len(splitted)):
            batch = list(splitted[b])
            if b >= self.complete_batches:
                incomplete_batch.extend(batch)
            else:
                complete_batches.append(batch)    
        self.batch_idx = complete_batches + [incomplete_batch]

    def __getitem__(self, idx):
        x_b, y_b = [], []
        # print(self.batch_idx[idx])
        for i in self.batch_idx[idx]:
            x, y = self.ds[i]
            x_b.append(x.unsqueeze(0))
            y_b.append(y)
        x_b = torch.cat(x_b, dim=0)
        y_b = torch.LongTensor(y_b)#.unsqueeze(1)
        return x_b, y_b

    def __len__(self):
        return self.num_batches

    def total_len(self):
        c = 0
        for b in range(len(self)):
            for i in self.batch_idx[b]:
                c += 1
        return c

class ContrastiveDataWrapper:
    def __init__(self, dataset, batch_size, num_views=1, *args, **kwargs):
        self.ds = dataset
        self.indices = list(range(len(self.ds)))
        self.num_batches = math.ceil(len(self.ds) / batch_size)
        self.batch_size = batch_size
        self.num_views = num_views
        self.complete_batches = len(self.ds) // batch_size
        self.incomplete_batches = len(self.ds) % batch_size

    def shuffle(self):
        idx = list(range(len(self.ds)))
        random.shuffle(idx)
        splitted = np.array_split(idx, self.num_batches)
        incomplete_batch = []
        complete_batches = []
        for b in range(0, len(splitted)):
            batch = list(splitted[b])
            if b >= self.complete_batches:
                incomplete_batch.extend(batch)
            else:
                complete_batches.append(batch)    
        self.batch_idx = complete_batches + [incomplete_batch]

    def __getitem__(self, idx):
        x_b, y_b = [], []
        # print(self.batch_idx[idx])
        for i in self.batch_idx[idx]:
            x_view, y_view = [], []
            for _ in range(self.num_views):
                x, y = self.ds[i]
                x_view.append(x.unsqueeze(0))
                y_view.append(y)
            x_view = torch.cat(x_view, dim=0)
            y_view = torch.LongTensor(y_view)
            x_b.append(x_view.unsqueeze(0))
            y_b.append(y_view.unsqueeze(0))
        x_b = torch.cat(x_b, dim=0)
        y_b = torch.cat(y_b, dim=0)[:, 0]
        return x_b, y_b

    def __len__(self):
        return self.num_batches

    def total_len(self):
        c = 0
        for b in range(len(self)):
            for i in self.batch_idx[b]:
                c += 1
        return c
    
if __name__ == "__main__":
    ds_size = 45000
    ds = torch.utils.data.TensorDataset(
        torch.randn(ds_size, 3, 32, 32),
        torch.randint(0, 10, (ds_size, )).int()
    )
    wrapped = DataWrapper(ds, 256)
    wrapped.shuffle()
    for ep in range(100):
        c = 0
        for b in range(len(wrapped)):
            x, y = wrapped[b]
            c += x.shape[0]
        assert(c == wrapped.total_len() == ds_size)
        wrapped.shuffle()
        assert(c == wrapped.total_len() == ds_size)

def split_dataset_disjoint_labels(num_clients, dataset, num_groups=2):
    assert(num_clients % num_groups == 0)
    
    labels = [y for x, y in dataset]
    labels_to_idx = {}
    for i, l in enumerate(labels):
        if l not in labels_to_idx:
            labels_to_idx[l] = []
        labels_to_idx[l].append(i)

    num_classes = len(set(labels))
    assert(num_classes % num_groups == 0)

    labels_to_chunks = {}
    chunk_counter = {}
    for k, v in labels_to_idx.items():
        labels_to_chunks[k] = np.split(np.array(v), num_clients // num_groups)
        chunk_counter[k] = 0

    group_ids = []
    for gid in range(num_groups):
        group_ids += [gid] * (num_clients // num_groups)
    random.shuffle(group_ids)

    unique_labels = list(range(num_classes))
    group_id_to_labels = {}
    random.shuffle(unique_labels)
    for i, l in enumerate(unique_labels):
        if i % num_groups not in group_id_to_labels:
            group_id_to_labels[i % num_groups] = []
        group_id_to_labels[i % num_groups].append(l)

    client_id_to_idx = {}
    for c in range(num_clients):
        for l in unique_labels:
            if l in group_id_to_labels[group_ids[c]]:
                if client_id_to_idx.get(c, False) == False:
                    client_id_to_idx[c] = [] 
                client_id_to_idx[c].extend(list(labels_to_chunks[l][chunk_counter[l]])) 
                chunk_counter[l] += 1
    return client_id_to_idx

def classwise_subset(total_dataset, num_clients, num_groups, test_split=0.1):
    client_id_to_idx = split_dataset_disjoint_labels(num_clients, total_dataset, num_groups)
    train, test = {}, {}
    train_sizes = torch.zeros((num_clients,))

    for c, indices in client_id_to_idx.items():
        train_idx, test_idx = train_test_split(indices, test_size=test_split)
        train[c] = train_idx
        test[c] = test_idx 

        train_sizes[c] = len(train_idx)
    return train, test, train_sizes