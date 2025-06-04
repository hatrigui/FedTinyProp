import pandas as pd
import os
import csv
from datetime import datetime

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
