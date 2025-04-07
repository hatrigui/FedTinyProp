import torch
import numpy as np

def compute_avg_grad_norm(model):
    """Compute average gradient norm across all parameters."""
    total_norm = 0.0
    num_params = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1
    return (total_norm / num_params) ** 0.5 if num_params > 0 else 0.0

def compute_adaptive_ratio(grad_norm, initial_grad_norm, phi_min=0.0):
    """Compute adaptive ratio based on gradient norms."""
    if initial_grad_norm == 0:
        return 1.0
    ratio = grad_norm / initial_grad_norm
    return max(phi_min, min(1.0, ratio))

def compute_sparsity_and_flops(model, full_flops_per_batch):
    """Compute model sparsity and FLOPS."""
    total_params = 0
    nonzero_params = 0
    for p in model.parameters():
        total_params += p.numel()
        nonzero_params += (p != 0).sum().item()
    
    sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    flops = full_flops_per_batch * (1 - sparsity)
    return sparsity, flops

def compute_sparse_deltas(model, initial_state, device):
    """Compute sparse weight deltas."""
    weight_deltas = {}
    peak_mem = 0
    for name, param in model.named_parameters():
        if name in initial_state:
            delta = param.detach().cpu() - initial_state[name]
            nonzero_indices = torch.nonzero(delta.abs() > 1e-6, as_tuple=True)
            if len(nonzero_indices) > 0:
                weight_deltas[name] = (nonzero_indices, delta[nonzero_indices])
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
    return weight_deltas, peak_mem 