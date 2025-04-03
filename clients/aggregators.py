def sparse_fedavg_aggregate(sparse_updates, global_model, model_name, tinyprop_params, dataset_sizes, **kwargs):
    total_samples = sum(dataset_sizes)
    global_state = global_model.state_dict()

    updated_state = {k: v.clone().float() for k, v in global_state.items()}

    for client_idx, sparse_dict in enumerate(sparse_updates):
        weight = dataset_sizes[client_idx] / total_samples
        for param_name, (values, indices) in sparse_dict.items():
            flat_param = updated_state[param_name].view(-1)
            flat_param.index_add_(0, indices, weight * values)
            updated_state[param_name] = flat_param.view_as(global_state[param_name])  
    global_model.load_state_dict(updated_state)
    return global_model
