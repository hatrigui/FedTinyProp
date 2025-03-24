import numpy as np
from torch.utils.data import Subset, random_split

def iid_partition(dataset, num_clients):
    """
    Partition dataset into IID splits.
    """
    total_len = len(dataset)
    lengths = [total_len // num_clients] * num_clients
    lengths[0] += total_len - sum(lengths)  # Adjust remainder
    subsets = random_split(dataset, lengths)
    return subsets

def non_iid_partition(dataset, num_clients=5, num_shards=10):
    """
    Partition dataset into non-IID splits using a shard-based method.
    """
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels)).T
    idxs_labels = idxs_labels[idxs_labels[:, 1].argsort()]
    sorted_idxs = idxs_labels[:, 0]
    shards = np.array_split(sorted_idxs, num_shards)
    np.random.shuffle(shards)
    shards_per_client = num_shards // num_clients
    client_idxs = []
    for i in range(num_clients):
        client_shards = shards[i*shards_per_client:(i+1)*shards_per_client]
        client_idxs.append(np.concatenate(client_shards))
    subsets = [Subset(dataset, indices.tolist()) for indices in client_idxs]
    return subsets

def dirichlet_partition(dataset, num_clients, alpha=0.5):
    """
    Partition a dataset among clients using a Dirichlet distribution to simulate label skew.
    """
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
    subsets = [Subset(dataset, indices) for indices in client_indices.values()]
    return subsets

def quantity_skew_partition(dataset, num_clients, min_ratio=0.1, max_ratio=0.5):
    """
    Partition dataset with quantity skew.
    Clients receive different amounts of data.
    """
    total_len = len(dataset)
    ratios = np.random.uniform(min_ratio, max_ratio, size=num_clients)
    ratios = ratios / ratios.sum()
    lengths = (ratios * total_len).astype(int)
    lengths[0] += total_len - lengths.sum()
    subsets = random_split(dataset, lengths.tolist())
    return subsets

def temporal_partition(dataset, num_clients):
    """
    Partition the dataset based on temporal order (sequential slices).
    """
    total_len = len(dataset)
    lengths = [total_len // num_clients] * num_clients
    lengths[0] += total_len - sum(lengths)
    indices = np.arange(total_len)
    subsets = []
    start = 0
    for l in lengths:
        subsets.append(Subset(dataset, indices[start:start+l]))
        start += l
    return subsets

def hybrid_partition(dataset, num_clients, alpha=0.5, min_ratio=0.1, max_ratio=0.5):
    """
    Hybrid partition: combine Dirichlet label skew with quantity skew.
    """
    dirichlet_subsets = dirichlet_partition(dataset, num_clients, alpha)
    hybrid_subsets = []
    for subset in dirichlet_subsets:
        total = len(subset)
        fraction = np.random.uniform(min_ratio, max_ratio)
        num_samples = int(total * fraction)
        if hasattr(subset, 'indices'):
            indices = subset.indices
        else:
            indices = np.arange(total)
        selected = np.random.choice(indices, size=num_samples, replace=False)
        hybrid_subsets.append(Subset(dataset, selected.tolist()))
    return hybrid_subsets
