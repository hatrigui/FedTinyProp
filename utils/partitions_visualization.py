import numpy as np
import matplotlib.pyplot as plt

def plot_class_distribution(client_datasets, dataset, title, num_classes=10):
    
    distributions = []
    for ds in client_datasets:
        indices = ds.indices if hasattr(ds, 'indices') else list(range(len(ds)))
        labels = np.array(dataset.targets)[indices]
        counts = np.bincount(labels, minlength=num_classes)
        distributions.append(counts)
        
    distributions = np.array(distributions)
    x = np.arange(num_classes)
    width = 0.15
    plt.figure(figsize=(10, 5))
    for i, dist in enumerate(distributions):
        plt.bar(x + i * width, dist, width, label=f'Client {i+1}')
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.legend()
    plt.show()
