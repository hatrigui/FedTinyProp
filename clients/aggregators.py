import torch
import numpy as np
from typing import List, Dict, Optional
from models.model import get_tinyprop_model

def sparse_fedavg_aggregate(sparse_updates, global_model, model_name, tinyprop_params, **kwargs):
    """Aggregate sparse updates from clients using FedAvg with quantization-aware aggregation."""
    updated_state = global_model.state_dict()
    
    # Calculate model size once
    model_size_bytes = sum(p.numel() * 4 for p in global_model.parameters())
    
    stats = {
        "total_updates": len(sparse_updates),
        "skipped_params": 0,
        "updated_params": 0,
        "total_params": sum(p.numel() for p in global_model.parameters()),
        "communication_bytes": model_size_bytes * len(sparse_updates),  # Initialize with download cost
        "layer_communication": {},
        "quantization_error": 0.0,
        "dense_updates": 0,
        "sparse_updates": 0,
        "model_size_bytes": model_size_bytes,
        "download_bytes": model_size_bytes * len(sparse_updates),  # Total download for all clients
        "upload_bytes": 0  # Will accumulate upload bytes
    }
    
    # Get client weights
    dataset_sizes = kwargs.get("dataset_sizes", [1.0] * len(sparse_updates))
    total_size = sum(dataset_sizes)
    client_weights = [size / total_size for size in dataset_sizes]
    
    print("\n[Server Debug] Starting aggregation of client updates...")
    
    for client_idx, (update, client_weight) in enumerate(zip(sparse_updates, client_weights)):
        for param_name, param in global_model.named_parameters():
            if param_name not in update:
                stats["skipped_params"] += 1
                continue
                
            try:
                update_data = update[param_name]
                if not isinstance(update_data, tuple):
                    # Handle dense update
                    if isinstance(update_data, torch.Tensor):
                        param.data += update_data * client_weight
                        stats["dense_updates"] += 1
                        layer_comm = param.numel() * 4
                        stats["communication_bytes"] += layer_comm
                        stats["upload_bytes"] += layer_comm
                        if param_name not in stats["layer_communication"]:
                            stats["layer_communication"][param_name] = 0
                        stats["layer_communication"][param_name] += layer_comm
                    continue
                
                indices, values = update_data
                if not isinstance(indices, torch.Tensor) or not isinstance(values, torch.Tensor):
                    continue
                    
                flat_param = param.data.view(-1)
                
                # Calculate communication cost using same method as client
                max_index = param.numel() - 1
                if max_index <= 255:  # uint8
                    index_bytes = 1
                elif max_index <= 65535:  # uint16
                    index_bytes = 2
                else:  # uint32
                    index_bytes = 4
                
                # Only count significant updates
                mask = values.abs() > 1e-4  # Match client threshold
                indices = indices[mask]
                values = values[mask]
                
                if indices.numel() == 0:
                    continue
                
                # Calculate bytes needed for this layer's update
                indices_bytes = indices.numel() * index_bytes
                values_bytes = values.numel() * 4  # float32 = 4 bytes
                format_overhead = 3  # Match client overhead
                
                # Calculate total bytes for this layer
                layer_comm = indices_bytes + values_bytes + format_overhead
                
                # Only use sparse if it's more efficient than dense
                dense_layer_bytes = param.numel() * 4
                if layer_comm >= dense_layer_bytes:
                    # Fall back to dense format
                    layer_comm = dense_layer_bytes
                    update_data = (None, param.data.cpu())
                
                # Add the weighted update
                if indices.numel() > 0:
                    flat_param.index_add_(0, indices, client_weight * values)
                    updated_state[param_name] = flat_param.view_as(param)
                    stats["updated_params"] += 1
                    stats["communication_bytes"] += layer_comm
                    stats["upload_bytes"] += layer_comm
                    stats["sparse_updates"] += 1
                    
                    # Track per-layer communication
                    if param_name not in stats["layer_communication"]:
                        stats["layer_communication"][param_name] = 0
                    stats["layer_communication"][param_name] += layer_comm
                
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
    print(f"Dense updates: {stats['dense_updates']}")
    print(f"Sparse updates: {stats['sparse_updates']}")
    print(f"Model size: {stats['model_size_bytes']/1024:.2f}KB")
    print(f"Total download: {stats['download_bytes']/1024:.2f}KB")
    print(f"Total upload: {stats['upload_bytes']/1024:.2f}KB")
    print(f"Total communication: {stats['communication_bytes']/1024:.2f}KB")
    print("\nPer-layer communication:")
    for layer, comm in stats["layer_communication"].items():
        print(f"  {layer}: {comm/1024:.2f}KB")
    
    global_model.load_state_dict(updated_state)
    return global_model, stats



def standard_fedavg_aggregate(client_params, global_model, dataset_sizes=None, **kwargs):
    """Aggregate dense updates from clients using standard FedAvg."""
    print("\n[Server Debug] Starting dense aggregation...")

    global_state = global_model.state_dict()
    tensor_keys = [k for k, v in global_state.items() if isinstance(v, torch.Tensor)]
    
    # Initialize parameter accumulator and stats
    aggregated = {k: torch.zeros_like(global_state[k]) for k in tensor_keys}
    stats = {
        "total_updates": len(client_params),
        "communication_bytes": sum(p.numel() * 4 for p in global_model.parameters()),  # 4 bytes per parameter
        "layer_communication": {},
        "updated_params": sum(p.numel() for p in global_model.parameters()),
        "skipped_params": 0
    }
    
    total_samples = sum(dataset_sizes) if dataset_sizes else len(client_params)
    
    for client_idx, params in enumerate(client_params):
        weight = dataset_sizes[client_idx] / total_samples if dataset_sizes else 1.0 / len(client_params)
        
        for k, param_arr in zip(tensor_keys, params):
            param_tensor = torch.from_numpy(param_arr).to(aggregated[k].device)
            aggregated[k] += param_tensor * weight

    # Load aggregated parameters back into model
    new_state = global_model.state_dict()
    for k in tensor_keys:
        new_state[k] = aggregated[k]
    global_model.load_state_dict(new_state, strict=False)

    return global_model, stats


