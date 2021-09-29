import numpy as np

comm_dict = {}
def compute_comm_cost(tensor):
    memo_key = str(list(tensor.shape))
    if memo_key in comm_dict:
        return comm_dict[memo_key]

    s = np.prod(tensor.size()) * 4 * 1e-6
    comm_dict[memo_key] = s
    return s