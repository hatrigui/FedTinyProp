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
            
            # Ensure each client gets at least one sample from each class
            if len(idx_k) < num_clients:
                raise ValueError(f"Not enough samples of class {k} ({len(idx_k)}) for {num_clients} clients")
            
            # First assign one sample to each client
            for i in range(num_clients):
                idx_batch[i].append(idx_k[i])
            
            # Then distribute remaining samples using Dirichlet distribution
            remaining_idx = idx_k[num_clients:]
            if len(remaining_idx) > 0:
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = (np.cumsum(proportions) * len(remaining_idx)).astype(int)[:-1]
                idx_split = np.split(remaining_idx, proportions)
                
                for i in range(num_clients):
                    idx_batch[i].extend(idx_split[i].tolist())

        sizes = [len(idx) for idx in idx_batch]
        
        if all(size >= min_size_per_client for size in sizes):
            break
        
        retries += 1
        if retries >= max_retries:
            raise ValueError(f"Could not satisfy min_size {min_size_per_client} after {max_retries} retries.")
    
    return [Subset(dataset, idxs) for idxs in idx_batch]
