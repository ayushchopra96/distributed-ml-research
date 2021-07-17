import torch
from triplettorch import TripletDataset


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
        return out_x, out_y, out_xc, out_yc


def collate_fn(batches):
    xb, yb, xcb, ycb = [], [], [], []

    for x, y, xc, yc in batches:

        xb.append(x.unsqueeze(0))
        yb.append(y.unsqueeze(0))

        xcb.append(xc)
        ycb.append(yc)

    xb = torch.cat(xb, dim=0)
    yb = torch.cat(yb, dim=0)

    xcb = torch.cat(xcb, dim=0)
    ycb = torch.cat(ycb, dim=0)

    return xb, yb, xcb, ycb
