import torch
from models.model import get_tinyprop_model


def weighted_aggregate(
    models,
    model_name,
    tinyprop_params,
    dataset_sizes=None
):
    if dataset_sizes is None:
        raise ValueError("Weighted aggregator needs dataset_sizes")

    

    global_model = get_tinyprop_model(model_name, tinyprop_params)
    global_dict = global_model.state_dict()

    total_samples = sum(dataset_sizes)
    for key in global_dict.keys():
        accum = torch.zeros_like(global_dict[key], dtype=torch.float)
        for model, size in zip(models, dataset_sizes):
            accum += model.state_dict()[key] * (size / total_samples)
        global_dict[key] = accum

    global_model.load_state_dict(global_dict)
    return global_model
