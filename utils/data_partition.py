import numpy as np
from torch.utils.data import Subset

def dirichlet_partition(dataset, num_clients, alpha, min_size_per_client=10, max_retries=100):
    labels = np.array(dataset.targets)
    K = len(np.unique(labels))
    retries = 0
    
    while True:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Normalize and convert to indexes
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_split = np.split(idx_k, proportions)
            
            for i in range(num_clients):
                idx_batch[i] += idx_split[i].tolist()

        sizes = [len(idx) for idx in idx_batch]
        
        if all(size >= min_size_per_client for size in sizes):
            break
        
        retries += 1
        if retries >= max_retries:
            raise ValueError(f"Could not satisfy min_size {min_size_per_client} after {max_retries} retries.")
    
    return [Subset(dataset, idxs) for idxs in idx_batch]
