import torch

def sparse_fedavg_aggregate(sparse_updates, global_model, model_name, tinyprop_params, dataset_sizes, **kwargs):
    """Aggregate sparse updates from clients using FedAvg with quantization-aware aggregation."""
    total_samples = sum(dataset_sizes)
    if total_samples == 0:
        print("[Server Debug] Warning: Total samples is 0, skipping aggregation")
        return global_model
        
    global_state = global_model.state_dict()
    updated_state = {k: v.clone().float() for k, v in global_state.items()}
    
    stats = {
        "total_updates": len(sparse_updates),
        "skipped_params": 0,
        "total_params": 0,
        "updated_params": 0,
        "communication_bytes": 0,
        "quantization_error": 0.0
    }
    
    print("\n[Server Debug] Starting aggregation of client updates...")
    
    for client_idx, sparse_dict in enumerate(sparse_updates):
        if dataset_sizes[client_idx] == 0:
            print(f"[Server Debug] Warning: Client {client_idx} has 0 samples, skipping")
            continue
            
        weight = dataset_sizes[client_idx] / total_samples
        print(f"\n[Server Debug] Processing client {client_idx} (weight: {weight:.4f})")
        
        for param_name, update in sparse_dict.items():
            if param_name not in updated_state:
                print(f"[Server Debug] Skipping unknown parameter: {param_name}")
                stats["skipped_params"] += 1
                continue
            
            try:
                indices, values = update
                stats["total_params"] += 1
                
                # Skip empty updates
                if isinstance(indices, torch.Tensor) and indices.numel() == 0:
                    print(f"[Server Debug] Empty update for parameter: {param_name}")
                    continue
                elif isinstance(indices, tuple) and all(idx.numel() == 0 for idx in indices):
                    print(f"[Server Debug] Empty update for parameter: {param_name}")
                    continue
                
                param = updated_state[param_name]
                flat_param = param.view(-1)
                
                # Handle tuple indices
                if isinstance(indices, tuple):
                    if len(indices) != 2:
                        print(f"[Server Debug] Invalid tuple indices length for parameter: {param_name}")
                        continue
                    batch_idx, param_idx = indices
                    if batch_idx.numel() == 0 or param_idx.numel() == 0:
                        print(f"[Server Debug] Empty tuple indices for parameter: {param_name}")
                        continue
                    # Convert tuple indices to flat indices
                    indices = param_idx.to(flat_param.device).to(torch.int64)
                else:
                    # Ensure indices and values are on the correct device and have correct types
                    indices = indices.to(flat_param.device).to(torch.int64)
                
                values = values.to(flat_param.device).to(torch.float32)
                
                # Skip if indices are invalid
                if indices.numel() > 0 and indices.max() >= flat_param.numel():
                    print(f"[Server Debug] Invalid indices for parameter: {param_name}")
                    continue
                
                # Handle scale factor for quantization
                scale_factor = getattr(update, 'scale_factor', 1.0)
                if scale_factor == 0:
                    print(f"[Server Debug] Warning: Scale factor is 0 for parameter: {param_name}, using 1.0")
                    scale_factor = 1.0
                dequantized_values = values / scale_factor
                
                # Ensure shapes match before adding
                if indices.numel() > 0 and indices.shape[0] != dequantized_values.shape[0]:
                    print(f"[Server Debug] Shape mismatch for parameter: {param_name}")
                    continue
                
                # Add the weighted update
                if indices.numel() > 0:
                    flat_param.index_add_(0, indices, weight * dequantized_values)
                    updated_state[param_name] = flat_param.view_as(param)
                    stats["updated_params"] += 1
                    
                    # Update communication statistics
                    indices_bytes = indices.numel() * indices.element_size()
                    values_bytes = values.numel() * values.element_size()
                    stats["communication_bytes"] += indices_bytes + values_bytes
                    
                    if hasattr(update, 'quantization_error'):
                        stats["quantization_error"] += update.quantization_error
                
            except Exception as e:
                print(f"[Server Debug] Error processing parameter {param_name}: {str(e)}")
                continue
    
    if stats["updated_params"] > 0:
        stats["quantization_error"] /= stats["updated_params"]
    else:
        print("[Server Debug] Warning: No parameters were updated during aggregation")
    
    print("\n[Server Debug] Aggregation Statistics:")
    print(f"Total clients processed: {stats['total_updates']}")
    print(f"Parameters skipped: {stats['skipped_params']}")
    print(f"Parameters updated: {stats['updated_params']}/{stats['total_params']}")
    print(f"Total communication: {stats['communication_bytes']/1024:.1f}KB")
    print(f"Average quantization error: {stats['quantization_error']:.6f}")
    
    global_model.load_state_dict(updated_state)
    return global_model