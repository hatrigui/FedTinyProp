import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_dataset(dataset_name, root='./data'):
    """
    Load a dataset given its name.
    Returns:
        (trainset, testset): Tuple of training and test datasets.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in ['mnist', 'fashionmnist']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if dataset_name == 'mnist':
            trainset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
            testset  = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        else:
            trainset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
            testset  = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
            
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        testset  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        testset  = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    print(f"[INFO] Loaded dataset '{dataset_name}' with {len(trainset)} training samples and {len(testset)} testing samples.")
    
    return trainset, testset
