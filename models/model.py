import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tinyProp import TinyPropParams, TinyPropConv2d, TinyPropLinear

DEFAULT_PARAMS = TinyPropParams(S_min=0.05, S_max=0.5, zeta=0.25, number_of_layers=2)
# For MNIST/FashionMNIST
class TinyPropCNN(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams = DEFAULT_PARAMS):
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

# For CIFAR10/CIFAR100
class TinyPropBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tinyPropParams, layer_number, downsample=False):
        super(TinyPropBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = TinyPropConv2d(in_channels, out_channels, kernel_size=3, padding=1,
                                    stride=stride, tinyPropParams=tinyPropParams, layer_number=layer_number)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = TinyPropConv2d(out_channels, out_channels, kernel_size=3, padding=1,
                                    tinyPropParams=tinyPropParams, layer_number=layer_number)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class TinyPropResNet8_CIFAR10(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams):
        super().__init__()
        self.conv = TinyPropConv2d(3, 16, kernel_size=3, padding=1, tinyPropParams=tinyprop_params, layer_number=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = TinyPropBlock(16, 32, tinyprop_params, layer_number=2, downsample=True)
        self.layer2 = TinyPropBlock(32, 64, tinyprop_params, layer_number=2, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = TinyPropLinear(64, 10, tinyprop_params, layer_number=3)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TinyPropResNet8_CIFAR100(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams):
        super().__init__()
        self.conv = TinyPropConv2d(3, 16, kernel_size=3, padding=1, tinyPropParams=tinyprop_params, layer_number=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = TinyPropBlock(16, 32, tinyprop_params, layer_number=2, downsample=True)
        self.layer2 = TinyPropBlock(32, 64, tinyprop_params, layer_number=2, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = TinyPropLinear(64, 100, tinyprop_params, layer_number=3)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



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
        return TinyPropResNet8_CIFAR10(tinyprop_params)
    elif dataset_name == 'cifar100':
        return TinyPropResNet8_CIFAR100(tinyprop_params)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
