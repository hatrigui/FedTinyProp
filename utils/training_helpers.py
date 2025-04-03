
def compute_avg_grad_norm(model):
    total_gnorm = 0.0
    total_params = 0
    for param in model.parameters():
        if param.grad is not None:
            g = param.grad.data
            total_gnorm += g.norm(2).item()
            total_params += 1
    return total_gnorm / total_params if total_params > 0 else 0.0


def compute_adaptive_ratio(avg_gnorm, initial_grad_norm, phi_min):
    if initial_grad_norm < 1e-9:
        return phi_min
    phi = avg_gnorm / initial_grad_norm
    return max(phi, phi_min)


def compute_sparsity_and_flops(model, full_flops_per_batch):
    delta_elements = 0
    total_elements = 0
    for param in model.parameters():
        if param.grad is not None:
            flat_grad = param.grad.data.view(-1)
            delta_elements += (flat_grad.abs() > 1e-9).sum().item()
            total_elements += flat_grad.numel()

    sparsity = 1 - (delta_elements / total_elements) if total_elements > 0 else 1.0
    flops = full_flops_per_batch * (1 - sparsity)
    return sparsity, flops


def compute_sparse_deltas(model, initial_state, device):
    sparse_deltas = {}
    total_nonzero_deltas = 0
    total_deltas = 0

    for name, param in model.state_dict().items():
        delta = param.detach().cpu() - initial_state[name]
        flat_delta = delta.view(-1)
        mask = flat_delta.abs() > 1e-9
        indices = mask.nonzero(as_tuple=False).reshape(-1)
        values = flat_delta[indices]
        sparse_deltas[name] = (values.to(device), indices.to(device))

        total_nonzero_deltas += len(indices)
        total_deltas += flat_delta.numel()

    sparsity = total_nonzero_deltas / total_deltas if total_deltas > 0 else 0.0
    peak_mem = total_nonzero_deltas * 4.0
    return sparse_deltas, sparsity, peak_mem
