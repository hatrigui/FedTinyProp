import torch.nn as nn
from typing import Dict, Tuple
import torch
def estimate_model_memory(model: nn.Module, batch_size: int, input_shape: Tuple[int]) -> Dict[str, float]:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.grad is not None)

    activation_bytes = 0
    dummy_input = torch.randn((batch_size, *input_shape)).to(next(model.parameters()).device)
    hooks = []
    outputs = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            outputs.append(output)
        elif isinstance(output, (list, tuple)):
            outputs.extend([o for o in output if isinstance(o, torch.Tensor)])

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(dummy_input)

    for out in outputs:
        activation_bytes += out.numel() * out.element_size()

    for h in hooks:
        h.remove()

    total_bytes = param_bytes + grad_bytes + activation_bytes
    return {
        "param_MB": param_bytes / 1024 / 1024,
        "grad_MB": grad_bytes / 1024 / 1024,
        "activation_MB": activation_bytes / 1024 / 1024,
        "total_MB": total_bytes / 1024 / 1024
    }