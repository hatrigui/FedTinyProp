import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Union
from torch.utils.data import Subset, Dataset

def plot_class_distribution(partitions: Union[List[Subset], Dict[str, List[Subset]]], dataset: Dataset, title: str = "Class Distribution", ax=None):
    """Plot class distribution for a partition."""
    if isinstance(partitions, dict):
        partitions = list(partitions.values())

    if hasattr(partitions[0].dataset, 'targets'):
        if isinstance(partitions[0].dataset.targets, np.ndarray):
            targets = partitions[0].dataset.targets
        else:
            targets = np.array(partitions[0].dataset.targets)
    else:
        targets = np.array([partitions[0].dataset[i][1] for i in range(len(partitions[0].dataset))])

    num_classes = len(np.unique(targets))
    print(f"[Debug] Number of classes: {num_classes}")
    num_clients = len(partitions)
    print(f"[Debug] Number of clients: {num_clients}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    bottom = np.zeros(num_classes)
    width = 0.8 / num_clients

    for client_idx, subset in enumerate(partitions):
        print(f"[Debug] Processing client {client_idx} with {len(subset)} samples")
        client_targets = []
        for idx in subset.indices:
            try:
                client_targets.append(targets[idx])
            except IndexError:
                print(f"[Warning] Invalid index {idx} for client {client_idx}")
        client_targets = np.array(client_targets)
        print(f"[Debug] Client {client_idx} has {len(client_targets)} valid targets")

        class_counts = np.zeros(num_classes)
        for t in client_targets:
            class_counts[int(t)] += 1
        print(f"[Debug] Client {client_idx} class counts: {class_counts}")

        ax.bar(np.arange(num_classes) + client_idx * width, class_counts, width=width, label=f'Client {client_idx}', bottom=bottom)
        bottom += class_counts

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(np.arange(num_classes) + width * (num_clients - 1) / 2)
    ax.set_xticklabels(range(num_classes))
    ax.legend()
    return fig

def plot_partition_heatmap(partitions: Union[List[Subset], Dict[str, List[Subset]]], dataset: Dataset, title="Client-Class Distribution Heatmap"):
    if isinstance(partitions, dict):
        partitions = list(partitions.values())

    if hasattr(partitions[0].dataset, 'targets'):
        if isinstance(partitions[0].dataset.targets, np.ndarray):
            targets = partitions[0].dataset.targets
        else:
            targets = np.array(partitions[0].dataset.targets)
    else:
        targets = np.array([partitions[0].dataset[i][1] for i in range(len(partitions[0].dataset))])

    num_classes = len(np.unique(targets))
    print(f"[Debug] Number of classes: {num_classes}")
    num_clients = len(partitions)
    print(f"[Debug] Number of clients: {num_clients}")

    data_matrix = np.zeros((num_clients, num_classes), dtype=int)
    for client_idx, subset in enumerate(partitions):
        print(f"[Debug] Processing client {client_idx} with {len(subset)} samples")
        client_targets = []
        for idx in subset.indices:
            try:
                client_targets.append(targets[idx])
            except IndexError:
                print(f"[Warning] Invalid index {idx} for client {client_idx}")
        client_targets = np.array(client_targets)
        print(f"[Debug] Client {client_idx} has {len(client_targets)} valid targets")

        for class_idx in range(num_classes):
            data_matrix[client_idx, class_idx] = np.sum(client_targets == class_idx)
        print(f"[Debug] Client {client_idx} class counts: {data_matrix[client_idx]}")

    df = pd.DataFrame(data_matrix, columns=[f"Class {i}" for i in range(num_classes)])
    df.index = [f"Client {i}" for i in range(num_clients)]

    print(f"[Debug] Data matrix shape: {data_matrix.shape}")
    print(f"[Debug] Data matrix sum: {data_matrix.sum()}")

    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Sample Count"})
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Client")
    plt.tight_layout()
    plt.show()

def print_partition_stats(partitions: Dict[str, List[Subset]], dataset_name: str = ""):
    """Print statistics about the data distribution in partitions."""
    print(f"\n=== Partition Statistics for {dataset_name} ===")
    for name, partition in partitions.items():
        print(f"\n{name.upper()} Partition:")
        total_samples = 0
        client_stats = []
        for client_idx, client_data in enumerate(partition):
            num_samples = len(client_data)
            total_samples += num_samples
            client_stats.append((client_idx, num_samples))
        print(f"Total samples: {total_samples}")
        print("\nPer-client distribution:")
        for client_idx, num_samples in client_stats:
            if total_samples > 0:
                percentage = (num_samples / total_samples) * 100
                print(f"Client {client_idx}: {num_samples} samples ({percentage:.2f}%)")
            else:
                print(f"Client {client_idx}: {num_samples} samples (0.00%)")

def plot_data_distribution(partitions: Dict[str, List[Subset]], dataset_name: str = ""):
    """Create a comprehensive visualization of the data distribution."""
    num_partitions = len(partitions)
    fig, axes = plt.subplots(2, num_partitions, figsize=(15, 10))
    if num_partitions == 1:
        axes = axes.reshape(2, 1)

    for idx, (name, partition) in enumerate(partitions.items()):
        plot_class_distribution(partition, partition[0].dataset, title=f"Class Distribution - {name}", ax=axes[0, idx])

        client_samples = [len(client_data) for client_data in partition]
        total_samples = sum(client_samples)
        percentages = [samples/total_samples * 100 if total_samples > 0 else 0.0 for samples in client_samples]

        ax = axes[1, idx]
        bars = ax.bar(range(len(client_samples)), client_samples)
        ax.set_xlabel('Client')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Sample Distribution - {name}')
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.suptitle(f"Data Distribution Analysis - {dataset_name}")
    plt.tight_layout()
    return fig