import torch
from models.model import get_tinyprop_model

def fedavg_aggregate(models, model_name, tinyprop_params, **kwargs):
    """
    FedAvg aggregator: simple average of parameters.
    """
    global_model = get_tinyprop_model(model_name, tinyprop_params)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.mean(torch.stack(
            [m.state_dict()[key].float() for m in models]), dim=0)

    global_model.load_state_dict(global_dict)
    return global_model

def fedprox_aggregate(models, model_name, tinyprop_params, dataset_sizes, mu=0.01, global_params=None, **kwargs):
    """
    FedProx aggregator (weighted average with proximal term).
    """
    if global_params is None:
        raise ValueError("FedProx requires the previous global model parameters.")

    global_model = get_tinyprop_model(model_name, tinyprop_params)
    global_dict = global_model.state_dict()

    total_samples = sum(dataset_sizes)
    for key in global_dict.keys():
        accum = torch.zeros_like(global_dict[key], dtype=torch.float)
        for model, size in zip(models, dataset_sizes):
            client_param = model.state_dict()[key]
            global_param = global_params[key]
            # Proximal term
            accum += (client_param - mu * (client_param - global_param)) * (size / total_samples)
        global_dict[key] = accum

    global_model.load_state_dict(global_dict)
    return global_model

def fednova_aggregate(models, model_name, tinyprop_params, dataset_sizes, local_steps, **kwargs):
    """
    Simplified FedNova aggregator: normalizes updates by local steps.
    """
    global_model = get_tinyprop_model(model_name, tinyprop_params)
    global_dict = global_model.state_dict()

    total_weight = sum(dataset_sizes)
    for key in global_dict.keys():
        accum = torch.zeros_like(global_dict[key], dtype=torch.float)
        for model, size, steps in zip(models, dataset_sizes, local_steps):
            weight = size / total_weight
            accum += (model.state_dict()[key] * weight) / steps  # normalization by local steps
        global_dict[key] = accum

    global_model.load_state_dict(global_dict)
    return global_model
