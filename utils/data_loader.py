import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_dataset(dataset_name, root='./data'):
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
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

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

def get_data_loaders(trainset, testset, batch_size=128, num_workers=2):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader