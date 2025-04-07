import numpy as np
from torch.utils.data import Subset, random_split

def dirichlet_partition(dataset, num_clients, alpha=0.5):
   
    labels = np.array(dataset.targets)
    num_classes = np.max(labels) + 1
    idx_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}
    client_indices = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        indices = idx_by_class[c]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, split_points)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
    
    return [client_indices[i] for i in range(num_clients)]
