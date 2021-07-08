import torch
import random
import math
import numpy as np

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

if __name__ == "__main__":
    ds = torch.utils.data.TensorDataset(
        torch.randn(1024, 3, 32, 32),
        torch.randint(0, 10, (1024, )).int()
    )
    wrapped = DataWrapper(ds, 31)
    wrapped.shuffle()
    for ep in range(100):
        for b in range(len(wrapped)):
            x, y = wrapped[b]
        wrapped.shuffle()
