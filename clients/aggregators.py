import torch

def sparse_fedavg_aggregate(sparse_updates, global_model, model_name, tinyprop_params, dataset_sizes, **kwargs):
    """Aggregate sparse updates from clients using FedAvg with quantization-aware aggregation."""
    total_samples = sum(dataset_sizes)
    global_state = global_model.state_dict()
    updated_state = {k: v.clone().float() for k, v in global_state.items()}
    
    # Track aggregation statistics
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
        weight = dataset_sizes[client_idx] / total_samples
        print(f"\n[Server Debug] Processing client {client_idx} (weight: {weight:.4f})")
        
        for param_name, update in sparse_dict.items():
            if param_name not in updated_state:
                print(f"[Server Debug] Skipping unknown parameter: {param_name}")
                stats["skipped_params"] += 1
                continue
            
            indices, values = update
            stats["total_params"] += 1
            
            # Before applying update, check indices format and emptiness
            if isinstance(indices, tuple):
                if all(idx.numel() == 0 for idx in indices):
                    print(f"[Server Debug] Empty update for parameter: {param_name}")
                    continue
                elif isinstance(indices, torch.Tensor):
                    if indices.numel() == 0:
                        print(f"[Server Debug] Empty update for parameter: {param_name}")
                        continue
                else:
                    print(f"[Server Debug] Unknown indices type for parameter: {param_name}")
                    continue
            
            param = updated_state[param_name]
            flat_param = param.view(-1)
            
            indices = indices.to(flat_param.device).to(torch.int64)
            values = values.to(flat_param.device).to(torch.float32)
            
            if indices.max() >= flat_param.numel():
                print(f"[Server Debug] Invalid indices for parameter: {param_name}")
                continue
            
            # Get scale factor for this update
            scale_factor = getattr(update, 'scale_factor', 1.0)
            
            # Dequantize values before aggregation
            dequantized_values = values / scale_factor
            
            # Apply the sparse update with proper scaling
            flat_param.index_add_(0, indices, weight * dequantized_values)
            updated_state[param_name] = flat_param.view_as(param)
            
            # Update statistics
            stats["updated_params"] += 1
            # Communication cost includes indices (int32) and quantized values (int8)
            stats["communication_bytes"] += (indices.numel() * 4 + values.numel())  # 4 bytes for indices, 1 byte for quantized values
            
            # Track quantization error if available
            if hasattr(update, 'quantization_error'):
                stats["quantization_error"] += update.quantization_error
    
    # Average quantization error across all updates
    if stats["updated_params"] > 0:
        stats["quantization_error"] /= stats["updated_params"]
    
    # Log aggregation statistics
    print("\n[Server Debug] Aggregation Statistics:")
    print(f"Total clients processed: {stats['total_updates']}")
    print(f"Parameters skipped: {stats['skipped_params']}")
    print(f"Parameters updated: {stats['updated_params']}/{stats['total_params']}")
    print(f"Total communication: {stats['communication_bytes']/1024:.1f}KB")
    print(f"Average quantization error: {stats['quantization_error']:.6f}")
    
    # Update global model
    global_model.load_state_dict(updated_state)
    
    return global_model