import pandas as pd
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional

def save_training_logs_csv(filepath, accuracy, flops, mem, comm, sparsity, quantization_error=None, memory_saved=None, 
                         download_bytes=None, upload_bytes=None, compression_ratio=None, model_size_bytes=None):
    rounds = list(range(1, len(accuracy) + 1))
    timestamps = [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in rounds]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "round": rounds,
        "accuracy": accuracy,
        "flops": flops,
        "memory_bytes": mem,
        "communication_bytes": comm,
        "sparsity": sparsity,
        "quantization_error": quantization_error if quantization_error is not None else [0.0] * len(rounds),
        "memory_saved": memory_saved if memory_saved is not None else [0.0] * len(rounds),
        "download_bytes": download_bytes if download_bytes is not None else [0.0] * len(rounds),
        "upload_bytes": upload_bytes if upload_bytes is not None else [0.0] * len(rounds),
        "compression_ratio": compression_ratio if compression_ratio is not None else [1.0] * len(rounds),
        "model_size_bytes": model_size_bytes if model_size_bytes is not None else [0.0] * len(rounds)
    })
    
    # Add human-readable columns
    df["communication_KB"] = df["communication_bytes"] / 1024
    df["communication_MB"] = df["communication_bytes"] / (1024 * 1024)
    df["download_KB"] = df["download_bytes"] / 1024
    df["upload_KB"] = df["upload_bytes"] / 1024
    df["model_size_KB"] = df["model_size_bytes"] / 1024
    
    df.to_csv(filepath, index=False)
    print(f"[INFO] Training logs saved to {filepath}")

def append_to_training_log_csv(filepath, round_num, accuracy, flops, memory_bytes, communication_bytes, 
                             sparsity, avg_grad_norm=None, avg_phi=None, skipped_batches=None, 
                             effective_compute_ratio=None, compression_ratio=None, quantization_error=None, 
                             avg_scale_factor=None, memory_saved=None, download_bytes=None, upload_bytes=None,
                             model_size_bytes=None, proximal_loss=None):
    """Append a single round's metrics to the CSV file."""
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_num,
        "accuracy": accuracy,
        "flops": flops,
        "memory_bytes": memory_bytes,
        "communication_bytes": communication_bytes,
        "sparsity": sparsity,
        "avg_grad_norm": avg_grad_norm if avg_grad_norm is not None else 0.0,
        "avg_phi": avg_phi if avg_phi is not None else 0.0,
        "skipped_batches": skipped_batches if skipped_batches is not None else 0,
        "effective_compute_ratio": effective_compute_ratio if effective_compute_ratio is not None else 1.0,
        "compression_ratio": compression_ratio if compression_ratio is not None else 1.0,
        "quantization_error": quantization_error if quantization_error is not None else 0.0,
        "avg_scale_factor": avg_scale_factor if avg_scale_factor is not None else 1.0,
        "memory_saved": memory_saved if memory_saved is not None else 0.0,
        "download_bytes": download_bytes if download_bytes is not None else 0.0,
        "upload_bytes": upload_bytes if upload_bytes is not None else 0.0,
        "model_size_bytes": model_size_bytes if model_size_bytes is not None else 0.0,
        "proximal_loss": proximal_loss if proximal_loss is not None else 0.0
    }
    
    # Add human-readable columns
    metrics["communication_KB"] = metrics["communication_bytes"] / 1024
    metrics["communication_MB"] = metrics["communication_bytes"] / (1024 * 1024)
    metrics["download_KB"] = metrics["download_bytes"] / 1024
    metrics["upload_KB"] = metrics["upload_bytes"] / 1024
    metrics["model_size_KB"] = metrics["model_size_bytes"] / 1024
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    print(f"[INFO] Appended round {round_num} metrics to {filepath}")

def save_per_client_metrics_csv(filepath: str, round_num: int, clients: List[Any], agg_metrics: Dict[str, Any]) -> None:
    """
    Save per-client metrics to a CSV file, with an additional row for aggregated metrics.
    
    Args:
        filepath: Path to the CSV file to save metrics to
        round_num: Current round number
        clients: List of client objects with metrics
        agg_metrics: Dictionary of aggregated metrics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # List to store all client metrics plus aggregated metrics
    all_metrics = []
    
    # Collect metrics for each client
    for client_idx, client in enumerate(clients):
        # Get client metrics
        client_metrics = client.get_metrics()
        
        # Create a row for this client
        metrics_row = {
            "timestamp": timestamp,
            "round": round_num,
            "client_id": f"client_{client_idx}",  # Changed column name to client_id to ensure it's a separate column
            "accuracy": client.local_evaluate(client.test_loader),
            "flops": client_metrics.get("flops", 0.0),
            "memory_bytes": client_metrics.get("memory", 0.0),
            "memory_saved": client_metrics.get("memory_saved", 0.0),
            "communication_bytes": client_metrics.get("communication", 0.0),
            "sparsity": client_metrics.get("sparsity", 0.0),
            "skipped_batches": client_metrics.get("skipped_batches", 0),
            "download_bytes": client_metrics.get("download_bytes", 0.0),
            "upload_bytes": client_metrics.get("upload_bytes", 0.0),
            "model_size_bytes": client_metrics.get("model_size_bytes", 0.0),
            "compression_ratio": client_metrics.get("compression_ratio", 1.0),
            "smoothed_adaptivity_factor": client_metrics.get("smoothed_adaptivity_factor", None)
        }
        
        # Add human-readable columns
        metrics_row["communication_KB"] = metrics_row["communication_bytes"] / 1024
        metrics_row["communication_MB"] = metrics_row["communication_bytes"] / (1024 * 1024)
        metrics_row["download_KB"] = metrics_row["download_bytes"] / 1024
        metrics_row["upload_KB"] = metrics_row["upload_bytes"] / 1024
        metrics_row["model_size_KB"] = metrics_row["model_size_bytes"] / 1024
        
        all_metrics.append(metrics_row)
    
    # Add aggregated metrics row
    agg_row = {
        "timestamp": timestamp,
        "round": round_num,
        "client_id": "agg",  # Changed column name to client_id to ensure it's a separate column
        "accuracy": agg_metrics.get("accuracy", 0.0),
        "flops": agg_metrics.get("flops", 0.0),
        "memory_bytes": agg_metrics.get("memory", 0.0),
        "memory_saved": agg_metrics.get("memory_saved", 0.0),
        "communication_bytes": agg_metrics.get("communication", 0.0),
        "sparsity": agg_metrics.get("sparsity", 0.0),
        "skipped_batches": agg_metrics.get("skipped_batches", 0),
        "download_bytes": agg_metrics.get("download_bytes", 0.0),
        "upload_bytes": agg_metrics.get("upload_bytes", 0.0),
        "model_size_bytes": agg_metrics.get("model_size_bytes", 0.0),
        "compression_ratio": agg_metrics.get("compression_ratio", 1.0),
        "effective_compute_ratio": agg_metrics.get("effective_compute_ratio", 1.0),
        "smoothed_adaptivity_factor": agg_metrics.get("smoothed_adaptivity_factor", None)
    }
    
    # Add human-readable columns for aggregated metrics
    agg_row["communication_KB"] = agg_row["communication_bytes"] / 1024
    agg_row["communication_MB"] = agg_row["communication_bytes"] / (1024 * 1024)
    agg_row["download_KB"] = agg_row["download_bytes"] / 1024
    agg_row["upload_KB"] = agg_row["upload_bytes"] / 1024
    agg_row["model_size_KB"] = agg_row["model_size_bytes"] / 1024
    
    all_metrics.append(agg_row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Ensure column order with client_id as a separate column
    if 'client_id' in df.columns:
        # Move client_id to be the third column (after timestamp and round)
        cols = df.columns.tolist()
        cols.remove('client_id')
        cols.insert(2, 'client_id')  # Insert after timestamp and round
        df = df[cols]
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filepath)
    
    # Write to CSV
    df.to_csv(
        filepath,
        mode='a' if file_exists else 'w',
        header=not file_exists,
        index=False
    )
    
    print(f"[INFO] Saved per-client metrics for round {round_num} to {filepath}")
