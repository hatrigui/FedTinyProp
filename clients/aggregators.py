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
        "communication_bytes": 0,  # Will accumulate total communication
        "layer_communication": {},
        "quantization_error": 0.0,
        "dense_updates": 0,
        "sparse_updates": 0,
        "model_size_bytes": model_size_bytes,
        "download_bytes": model_size_bytes * len(sparse_updates),  # Total download for all clients
        "upload_bytes": 0,  # Will accumulate upload bytes
        "compression_ratio": 1.0  # Will be calculated after processing all updates
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
                if not isinstance(update_data, tuple) or len(update_data) != 2:
                    print(f"[Server Debug] Skipping malformed update for {param_name}: not a 2-tuple")
                    stats["skipped_params"] += 1
                    continue
                
                indices, values = update_data
                
                # Handle dense update (indices is None)
                if indices is None:
                    if not isinstance(values, torch.Tensor):
                        print(f"[Server Debug] Skipping malformed dense update for {param_name}: values not a tensor")
                        stats["skipped_params"] += 1
                        continue
                    
                    param.data += values.to(param.device) * client_weight
                    stats["dense_updates"] += 1
                    layer_comm = param.numel() * 4
                    stats["communication_bytes"] += layer_comm
                    stats["upload_bytes"] += layer_comm
                    if param_name not in stats["layer_communication"]:
                        stats["layer_communication"][param_name] = 0
                    stats["layer_communication"][param_name] += layer_comm
                    continue
                
                # Handle sparse update
                if not isinstance(indices, torch.Tensor) or not isinstance(values, torch.Tensor):
                    print(f"[Server Debug] Skipping malformed sparse update for {param_name}: invalid tensor types")
                    stats["skipped_params"] += 1
                    continue
                
                if indices.numel() == 0:
                    print(f"[Server Debug] Skipping empty sparse update for {param_name}")
                    stats["skipped_params"] += 1
                    continue
                
                # Calculate communication cost for sparse update
                max_index = param.numel() - 1
                index_bytes = 1 if max_index <= 255 else (2 if max_index <= 65535 else 4)
                indices_bytes = indices.numel() * index_bytes
                values_bytes = values.numel() * 4
                format_overhead = 3  # For storing format information
                layer_comm = indices_bytes + values_bytes + format_overhead
                
                # Apply the sparse update
                flat_param = param.data.view(-1)
                flat_param.index_add_(0, indices.to(param.device), values.to(param.device) * client_weight)
                updated_state[param_name] = flat_param.view_as(param)
                
                stats["updated_params"] += 1
                stats["communication_bytes"] += layer_comm
                stats["upload_bytes"] += layer_comm
                stats["sparse_updates"] += 1
                
                if param_name not in stats["layer_communication"]:
                    stats["layer_communication"][param_name] = 0
                stats["layer_communication"][param_name] += layer_comm
                
            except Exception as e:
                print(f"[Server Debug] Error processing parameter {param_name}: {str(e)}")
                stats["skipped_params"] += 1
                continue
    
    # Calculate final compression ratio
    if stats["upload_bytes"] > 0:
        stats["compression_ratio"] = (stats["model_size_bytes"] * len(sparse_updates)) / stats["upload_bytes"]
    
    # Update total communication
    stats["communication_bytes"] = stats["download_bytes"] + stats["upload_bytes"]
    
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
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print("\nPer-layer communication:")
    for layer, comm in stats["layer_communication"].items():
        print(f"  {layer}: {comm/1024:.2f}KB")
    
    global_model.load_state_dict(updated_state)
    return global_model, stats



