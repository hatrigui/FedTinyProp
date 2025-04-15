import pandas as pd
import os
import csv

def save_training_logs_csv(filepath, accuracy, flops, mem, comm, sparsity, quantization_error=None):
    rounds = list(range(1, len(accuracy) + 1))
    df = pd.DataFrame({
        "round": rounds,
        "accuracy": accuracy,
        "flops": flops,
        "memory_bytes": mem,
        "communication_bytes": comm,
        "sparsity": sparsity,
        "quantization_error": quantization_error if quantization_error is not None else [0.0] * len(rounds)
    })
    df.to_csv(filepath, index=False)
    print(f"[INFO] Training logs saved to {filepath}")

def append_to_training_log_csv(
    filepath,
    round_num,
    accuracy,
    flops,
    memory_bytes,
    communication_bytes,
    sparsity,
    avg_grad_norm,
    avg_phi,
    skipped_batches,
    effective_compute_ratio,
    compression_ratio,
    quantization_error=0.0,
    avg_scale_factor=1.0
):
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "round", "accuracy", "flops", "memory_bytes", "communication_bytes",
                "sparsity", "avg_grad_norm", "avg_phi", "skipped_batches",
                "effective_compute_ratio", "compression_ratio", "quantization_error",
                "avg_scale_factor"
            ])
        writer.writerow([
            round_num, accuracy, flops, memory_bytes, communication_bytes,
            sparsity, avg_grad_norm, avg_phi, skipped_batches,
            effective_compute_ratio, compression_ratio, quantization_error,
            avg_scale_factor
        ])
