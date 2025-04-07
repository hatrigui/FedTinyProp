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
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, tp_params, layer_number):
        super(BasicBlock, self).__init__()
        self.conv1 = TinyPropConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                     tinyPropParams=tp_params, layer_number=layer_number)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = TinyPropConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                     tinyPropParams=tp_params, layer_number=layer_number)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                TinyPropConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                               tinyPropParams=tp_params, layer_number=layer_number),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TinyPropResNet8(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams, num_classes=10):
        super(TinyPropResNet8, self).__init__()
        self.in_channels = 16

        self.conv1 = TinyPropConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                    tinyPropParams=tinyprop_params, layer_number=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, num_blocks=1, stride=1, tp_params=tinyprop_params, layer_number=1)
        self.layer2 = self._make_layer(32, num_blocks=1, stride=2, tp_params=tinyprop_params, layer_number=2)
        self.layer3 = self._make_layer(64, num_blocks=1, stride=2, tp_params=tinyprop_params, layer_number=3)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = TinyPropLinear(64, num_classes, tinyPropParams=tinyprop_params, layer_number=3)

    def _make_layer(self, out_channels, num_blocks, stride, tp_params, layer_number):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s, tp_params, layer_number))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



def get_tinyprop_model(dataset_name, tinyprop_params=None):
    """
    Return the appropriate TinyProp-based model based on the dataset name.
    """
    if tinyprop_params is None:
        tinyprop_params = TinyPropParams(S_min=0.1, S_max=0.9, zeta=0.95, number_of_layers=2)

    dataset_name = dataset_name.lower()

    if dataset_name in ['mnist', 'fashionmnist']:
        return TinyPropCNN(tinyprop_params)
    elif dataset_name == "cifar10":
        return TinyPropResNet8(tinyprop_params, num_classes=10)
    elif dataset_name == "cifar100":
        return TinyPropResNet8(tinyprop_params, num_classes=100)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
