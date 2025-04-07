import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union

def plot_class_distribution(partitions: Union[List[List[int]], Dict[int, List[int]]], dataset, title: str = "Class Distribution"):
 
    # Handle both numpy arrays and lists for targets
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, np.ndarray):
            targets = dataset.targets
        else:
            targets = np.array(dataset.targets)
    else:
        # If targets are not directly accessible, get them from the dataset
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(targets))
    
    # Convert partitions to list format if it's a dictionary
    if isinstance(partitions, dict):
        partitions = list(partitions.values())
    
    num_clients = len(partitions)
    
    # Create a figure with subplots for each client
    fig, axes = plt.subplots(1, num_clients, figsize=(15, 5))
    if num_clients == 1:
        axes = [axes]
    
    for client_idx, client_indices in enumerate(partitions):
        # Get the targets for this client
        client_targets = []
        for idx in client_indices:
            if isinstance(idx, (list, np.ndarray)):
                # Handle nested indices
                for sub_idx in idx:
                    client_targets.append(targets[sub_idx])
            else:
                client_targets.append(targets[idx])
        client_targets = np.array(client_targets)
        
        # Count occurrences of each class
        class_counts = np.zeros(num_classes)
        for t in client_targets:
            class_counts[t] += 1
        
        # Plot the distribution
        ax = axes[client_idx]
        ax.bar(range(num_classes), class_counts)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'Client {client_idx}')
        ax.set_xticks(range(num_classes))
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


