import torch
import torch.nn as nn
from typing import Dict, Tuple

def compute_conv2d_flops(input_shape: Tuple[int, int, int, int], 
                        out_channels: int, 
                        kernel_size: int, 
                        stride: int = 1, 
                        padding: int = 0, 
                        groups: int = 1) -> int:
    """Compute FLOPs for a Conv2d layer."""
    batch_size, in_channels, height, width = input_shape
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Each output element requires:
    # - kernel_size * kernel_size * in_channels / groups multiplications
    # - kernel_size * kernel_size * in_channels / groups - 1 additions
    flops_per_output = (2 * kernel_size * kernel_size * in_channels / groups - 1)
    # Calculate FLOPs for a single sample
    total_flops = out_channels * output_height * output_width * flops_per_output
    
    return int(total_flops)

def compute_linear_flops(input_features: int, output_features: int) -> int:
    """Compute FLOPs for a Linear layer."""
    # Each output element requires:
    # - input_features multiplications
    # - input_features - 1 additions
    flops_per_output = (2 * input_features - 1)
    # Calculate FLOPs for a single sample
    total_flops = output_features * flops_per_output
    
    return int(total_flops)

def compute_conv1d_flops(input_shape: Tuple[int, int, int], 
                        out_channels: int, 
                        kernel_size: int, 
                        stride: int = 1, 
                        padding: int = 0, 
                        groups: int = 1) -> int:
    """Compute FLOPs for a Conv1d layer."""
    batch_size, in_channels, length = input_shape
    output_length = (length + 2 * padding - kernel_size) // stride + 1
    
    # Each output element requires:
    # - kernel_size * in_channels / groups multiplications
    # - kernel_size * in_channels / groups - 1 additions
    flops_per_output = (2 * kernel_size * in_channels / groups - 1)
    # Calculate FLOPs for a single sample
    total_flops = out_channels * output_length * flops_per_output
    
    return int(total_flops)

def compute_batchnorm_flops(input_shape: Tuple[int, ...]) -> int:
    """Compute FLOPs for a BatchNorm layer."""
    # For each element:
    # - 1 multiplication (gamma)
    # - 1 addition (beta)
    # - 1 division (normalization)
    # - 1 subtraction (mean)
    flops_per_element = 4
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim
    return int(total_elements * flops_per_element)

def compute_pooling_flops(input_shape: Tuple[int, ...], kernel_size: int) -> int:
    """Compute FLOPs for a MaxPool or AvgPool layer."""
    # For each output element:
    # - kernel_size * kernel_size - 1 comparisons (max) or
    # - kernel_size * kernel_size - 1 additions (avg)
    flops_per_output = kernel_size * kernel_size - 1
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim
    return int(total_elements * flops_per_output)

def compute_layer_flops(layer: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Compute FLOPs for a single layer."""
    if isinstance(layer, nn.Conv2d):
        return compute_conv2d_flops(
            input_shape,
            layer.out_channels,
            layer.kernel_size[0],
            layer.stride[0],
            layer.padding[0],
            layer.groups
        )
    elif isinstance(layer, nn.Conv1d):
        return compute_conv1d_flops(
            input_shape,
            layer.out_channels,
            layer.kernel_size[0],
            layer.stride[0],
            layer.padding[0],
            layer.groups
        )
    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return compute_batchnorm_flops(input_shape)
    elif isinstance(layer, (nn.MaxPool2d, nn.MaxPool1d)):
        return compute_pooling_flops(input_shape, layer.kernel_size)
    elif isinstance(layer, (nn.AvgPool2d, nn.AvgPool1d)):
        return compute_pooling_flops(input_shape, layer.kernel_size)
    elif isinstance(layer, nn.Linear):
        return compute_linear_flops(input_shape[1], layer.out_features)
    return 0

def compute_model_flops(model: nn.Module, 
                        input_shape: Tuple[int, ...], 
                        layer_sparsity_dict: Dict[str, float] = None) -> Tuple[int, Dict[str, int]]:
    """
    Compute total FLOPs for the model with per-layer sparsity.

    Args:
        model: Neural network model.
        input_shape: Shape of input tensor (batch_size, channels, height, width).
        layer_sparsity_dict: Dictionary mapping layer names to their sparsity (0.0-1.0).

    Returns:
        total_flops: Adjusted total FLOPs.
        layer_flops: Dictionary of per-layer adjusted FLOPs.
    """
    layer_flops = {}
    total_flops = 0
    current_shape = input_shape
    batch_size = input_shape[0]

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d, 
                            nn.MaxPool2d, nn.MaxPool1d, nn.AvgPool2d, nn.AvgPool1d)):
            sparsity = layer_sparsity_dict.get(name, 0.0) if layer_sparsity_dict else 0.0
            layer_flop = compute_layer_flops(layer, current_shape)
            # Apply only sparsity, batch size will be applied by caller
            adjusted_flop = int(layer_flop * (1 - sparsity))

            layer_flops[name] = adjusted_flop
            total_flops += adjusted_flop

            # Update shape for next layer
            if isinstance(layer, nn.Conv2d):
                _, _, h, w = current_shape
                out_h = (h + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                out_w = (w + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                current_shape = (batch_size, layer.out_channels, out_h, out_w)
            elif isinstance(layer, nn.Conv1d):
                _, _, l = current_shape
                out_l = (l + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                current_shape = (batch_size, layer.out_channels, out_l)
            elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
                _, c, h, w = current_shape
                out_h = (h - layer.kernel_size) // layer.stride + 1
                out_w = (w - layer.kernel_size) // layer.stride + 1
                current_shape = (batch_size, c, out_h, out_w)
            elif isinstance(layer, (nn.MaxPool1d, nn.AvgPool1d)):
                _, c, l = current_shape
                out_l = (l - layer.kernel_size) // layer.stride + 1
                current_shape = (batch_size, c, out_l)
            elif isinstance(layer, nn.Linear):
                current_shape = (batch_size, layer.out_features)

    return total_flops, layer_flops

    