import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tinyProp import TinyPropParams, TinyPropConv2d, TinyPropLinear, TinyPropLayer, TinyPropConv1d

# Use number_of_layers=5 to match TinyProp1DCNN architecture
# Other models will override this with their specific layer counts
DEFAULT_PARAMS = TinyPropParams(S_min=0.05, S_max=0.5, zeta=0.25, number_of_layers=5)
# For MNIST/FashionMNIST
class TinyPropCNN(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams = DEFAULT_PARAMS, num_classes: int = 10):
        super(TinyPropCNN, self).__init__()
        
        # Initialize TinyPropLayer
        self.tpLayer = TinyPropLayer(tinyprop_params.number_of_layers)
        self.tpParams = tinyprop_params
        self.current_round = 0  # Initialize current_round
        
        self.conv1 = TinyPropConv2d(1, 32, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=1, padding=1)
        self.conv2 = TinyPropConv2d(32, 64, kernel_size=3, tinyPropParams=tinyprop_params, layer_number=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = TinyPropLinear(64 * 14 * 14, 128, tinyPropParams=tinyprop_params, layer_number=3)
        self.fc2 = TinyPropLinear(128, num_classes, tinyPropParams=tinyprop_params, layer_number=4)

    @property
    def phi_k(self):
        """Get the current phi_k value from the TinyPropLayer's history."""
        history = self.tpLayer.stats.get("phi_k_history", [])
        return history[-1] if history else 0.0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# For CIFAR10/CIFAR100
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, tinyprop_params=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TinyPropResNet8(nn.Module):
    def __init__(self, num_classes=10, tinyprop_params=None):
        super(TinyPropResNet8, self).__init__()
        self.in_planes = 32  # Reduced from 64
        self.tinyprop_params = tinyprop_params or DEFAULT_PARAMS
        
        # Initialize TinyPropLayer
        self.tpLayer = TinyPropLayer(self.tinyprop_params.number_of_layers)
        self.tpParams = self.tinyprop_params
        self.current_round = 0  # Initialize current_round

        # Initial convolution with smaller channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Three layers with reduced channels
        self.layer1 = self._make_layer(32, 1, stride=1)   # First layer: 32 channels
        self.layer2 = self._make_layer(64, 1, stride=2)   # Second layer: 64 channels
        self.layer3 = self._make_layer(128, 1, stride=2)  # Third layer: 128 channels
        
        # Final classification layer - fixed dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # This will give us 128 features
        self.linear = nn.Linear(128, num_classes)    # Input features = 128

    @property
    def phi_k(self):
        """Get the current phi_k value from the TinyPropLayer's history."""
        history = self.tpLayer.stats.get("phi_k_history", [])
        return history[-1] if history else 0.0

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, self.tinyprop_params))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)  # This will give us [batch_size, 128, 1, 1]
        out = out.view(out.size(0), -1)  # Flatten to [batch_size, 128]
        out = self.linear(out)
        return out

class TinyProp1DCNN(nn.Module):
    def __init__(self, tinyprop_params: TinyPropParams = DEFAULT_PARAMS, num_classes: int = 6):
        super(TinyProp1DCNN, self).__init__()
        
        # Initialize TinyPropLayer
        self.tpLayer = TinyPropLayer(tinyprop_params.number_of_layers)
        self.tpParams = tinyprop_params
        self.current_round = 0  # Initialize current_round
        
        # First convolutional block
        self.conv1 = TinyPropConv1d(1, 64, kernel_size=5, stride=1, padding=2, 
                                   tinyPropParams=tinyprop_params, layer_number=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Second convolutional block
        self.conv2 = TinyPropConv1d(64, 128, kernel_size=5, stride=1, padding=2,
                                   tinyPropParams=tinyprop_params, layer_number=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Third convolutional block
        self.conv3 = TinyPropConv1d(128, 256, kernel_size=5, stride=1, padding=2,
                                   tinyPropParams=tinyprop_params, layer_number=3)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = TinyPropLinear(256, 128, tinyPropParams=tinyprop_params, layer_number=4)
        self.fc2 = TinyPropLinear(128, num_classes, tinyPropParams=tinyprop_params, layer_number=5)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    @property
    def phi_k(self):
        """Get the current phi_k value from the TinyPropLayer's history."""
        history = self.tpLayer.stats.get("phi_k_history", [])
        return history[-1] if history else 0.0

    def forward(self, x):
        # Reshape input if needed (batch_size, channels, sequence_length)
        if len(x.shape) == 4:  # If input is (batch, 1, 24, 24)
            x = x.squeeze(1)  # Remove channel dimension
            x = x.view(x.size(0), 1, -1)  # Reshape to (batch, 1, 576)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove last dimension
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_tinyprop_model(dataset_name, tinyprop_params):
    """Get the appropriate TinyProp-based model for the dataset."""
    dataset_name = dataset_name.lower()
    
    if dataset_name in ["mnist", "fashionmnist"]:
        return TinyPropCNN(tinyprop_params=tinyprop_params, num_classes=10)
    elif dataset_name == "cifar10":
        return TinyPropResNet8(num_classes=10, tinyprop_params=tinyprop_params)
    elif dataset_name == "cifar100":
        return TinyPropResNet8(num_classes=100, tinyprop_params=tinyprop_params)
    elif dataset_name == "har":
        return TinyProp1DCNN(tinyprop_params=tinyprop_params, num_classes=6)  # 6 activity classes
    else:
        raise ValueError(f"No model defined for dataset: {dataset_name}")
