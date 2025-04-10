import numpy as np
from torch.utils.data import Subset, random_split

def dirichlet_partition(dataset, num_clients, alpha):
    from torch.utils.data import Subset
    labels = np.array(dataset.targets)
    idx_batch = [[] for _ in range(num_clients)]
    min_size = 0
    K = len(np.unique(labels))

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_split = np.split(idx_k, proportions)
            for i in range(num_clients):
                idx_batch[i] += idx_split[i].tolist()
        min_size = min(len(idx) for idx in idx_batch)

    subsets = [Subset(dataset, idx) for idx in idx_batch]
    return subsets

