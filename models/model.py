import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tinyProp import TinyPropParams, TinyPropConv2d, TinyPropLinear

tinyprop_params = TinyPropParams(S_min=0.1, S_max=0.9, zeta=0.95, number_of_layers=2)

class TinyPropCNN(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams):
        super(TinyPropCNN, self).__init__()
        self.conv1 = TinyPropConv2d(1, 32, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=1, padding=1)
        self.conv2 = TinyPropConv2d(32, 64, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = TinyPropLinear(64 * 14 * 14, 128, tinyPropParams=tinyprop_params, layer_number=2)
        self.fc2 = TinyPropLinear(128, 10, tinyPropParams=tinyprop_params, layer_number=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyPropCNN_CIFAR(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams):
        super(TinyPropCNN_CIFAR, self).__init__()
        self.conv1 = TinyPropConv2d(3, 32, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=1, padding=1)
        self.conv2 = TinyPropConv2d(32, 64, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = TinyPropLinear(64 * 16 * 16, 128, tinyPropParams=tinyprop_params, layer_number=2)
        self.fc2 = TinyPropLinear(128, 10, tinyPropParams=tinyprop_params, layer_number=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyPropCNN_CIFAR100(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams):
        super(TinyPropCNN_CIFAR100, self).__init__()
        self.conv1 = TinyPropConv2d(3, 32, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=1, padding=1)
        self.conv2 = TinyPropConv2d(32, 64, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = TinyPropLinear(64 * 16 * 16, 128, tinyPropParams=tinyprop_params, layer_number=2)
        self.fc2 = TinyPropLinear(128, 100, tinyPropParams=tinyprop_params, layer_number=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def get_tinyprop_model(dataset_name, tinyprop_params=None):
    """
    Return the appropriate TinyProp-based model based on the dataset name.
    """
    if tinyprop_params is None:
        tinyprop_params = TinyPropParams(S_min=0.1, S_max=0.9, zeta=0.95, number_of_layers=2)

    dataset_name = dataset_name.lower()

    if dataset_name in ['mnist', 'fashionmnist']:
        return TinyPropCNN(tinyprop_params)
    elif dataset_name == 'cifar10':
        return TinyPropCNN_CIFAR(tinyprop_params)
    elif dataset_name == 'cifar100':
        return TinyPropCNN_CIFAR100(tinyprop_params)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
