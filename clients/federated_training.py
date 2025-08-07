from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from clients.rigl_client import FederatedRigLClient
from models.model import get_tinyprop_model 
from models.config import get_tinyprop_config, get_dense_config
from utils.early_stopping import EarlyStoppingMonitor
from utils.save_results import append_to_training_log_csv, save_training_logs_csv
from utils.adaptive_sparsification import AdaptiveSparsifier
from models.model import get_tinyprop_model
from clients.aggregators import sparse_fedavg_aggregate
from utils.adaptive_sparsification import AdaptiveSparsifier
from models.tinyProp import get_phi_k
import numpy as np
import random
import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

def evaluate_model(model, data_loader):
    """Evaluate model accuracy on a given data loader."""
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def federated_training(
    client_datasets,
    model_name,
    testset,
    tinyprop_params,
    aggregator_fn,
    aggregator_kwargs=None,
    rounds=200,
    device="cpu",
    local_epochs=1,
    early_stopping_patience=5,
    early_stopping_delta=0.0001,
    csv_log_path=None,
    initial_sparsity: float = 0.3,
    target_sparsity: float = 0.9,
    energy_budget: Optional[float] = None,
    num_clients=None,
    num_rounds=None,
    partition_type=None,
    alpha=None,
    seed: int = 42,
    save_dir=None,
    save_interval=1,
    quantization_bits=8,
    use_dense_baseline: bool = False,
    use_fedprox: bool = False,
    use_fedprune: bool = False,
    use_rigl: bool = False,
    fedprox_mu: float = 0.1,
    fedprune_sparsity: float = 0.5,
    rigl_initial_sparsity: float = 0.5,
    rigl_target_sparsity: float = 0.9,
    rigl_update_interval: int = 100,
    rigl_final_update_epoch: int = 100,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if aggregator_kwargs is None:
        aggregator_kwargs = {}

    metrics_log = {
        "timestamp": [],
        "round": [],
        "accuracy": [],
        "flops": [],
        "rigl_sparsity": [],  # Always initialize as empty list
        "memory": [],
        "memory_saved": [],
        "communication": [],
        "sparsity": [],
        "skipped_batches": [],
        "effective_compute_ratio": [],
        "compression_ratio": [],
        "download_bytes": [],
        "upload_bytes": [],
        "model_size_bytes": [],
        "communication_KB": [],
        "communication_MB": [],
        "download_KB": [],
        "upload_KB": [],
        "model_size_KB": [],
        "client_eval_history": []
    }

    # Use dense config if specified
    if use_dense_baseline:
        config = get_dense_config(model_name)
        tinyprop_params = config["tinyprop_params"]
    else:
        config = get_tinyprop_config(model_name)
        tinyprop_params = config["tinyprop_params"]

    global_model = get_tinyprop_model(model_name, tinyprop_params).to(device)

    # Initialize clients
    clients = []
    dataset_sizes = []
    
    print("\n[Training Debug] Initializing clients...")
    for i, dataset in enumerate(client_datasets):
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(testset, batch_size=32, shuffle=False)
        dataset_sizes.append(len(train_loader.dataset))
        
        client_model = get_tinyprop_model(model_name, tinyprop_params).to(device)
        
        # Apply FedPrune static masks if enabled
        if use_fedprune:
            for param in client_model.parameters():
                mask = (torch.rand_like(param) > fedprune_sparsity).float().to(device)
                param.data.mul_(mask)
                param.register_hook(lambda grad, mask=mask: grad.mul_(mask))
        
        client_cfg = config.copy()
        client_cfg["use_fedprox"] = use_fedprox
        client_cfg["fedprox_mu"] = fedprox_mu
        
        if use_rigl:
            client = FederatedRigLClient(
                client_id=i,
                model=client_model,
                train_loader=train_loader,
                test_loader=test_loader,
                cfg=client_cfg,
                device=device,
                dataset_name=model_name,
                rigl_initial_sparsity=rigl_initial_sparsity,
                rigl_target_sparsity=rigl_target_sparsity,
                rigl_update_interval=rigl_update_interval,
                rigl_final_update_epoch=rigl_final_update_epoch
            )
        else:
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_loader=train_loader,
                test_loader=test_loader,
                cfg=client_cfg,
                device=device,
                dataset_name=model_name
            )
        clients.append(client)
        print(f"[Training Debug] Client {i} initialized with {dataset_sizes[i]} samples")

    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    early_stopper = None if early_stopping_patience is None or early_stopping_patience <= 0 else EarlyStoppingMonitor(patience=early_stopping_patience, delta=early_stopping_delta)
    sparsifier = AdaptiveSparsifier(initial_sparsity, target_sparsity, rounds, energy_budget=energy_budget)
    history = {k: [] for k in ["train_loss", "train_accuracy", "test_accuracy", "sparsity", "energy", "communication"]}

    # Initialize quantization metrics
    quantization_errors = []
    avg_scale_factors = []

    # Initialize consolidated metrics with timestamp
    consolidated_metrics = {
        "timestamp": [],
        "round": [],
        "accuracy": [],
        "flops": [],
        "memory": [],
        "memory_saved": [],
        "communication": [],
        "sparsity": [],
        "skipped_batches": [],
        "effective_compute_ratio": [],
        "compression_ratio": [],
        "download_bytes": [],
        "upload_bytes": [],
        "model_size_bytes": [],
        "quantization_error": [],
        "avg_scale_factor": [],
        "method": []
    }

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_deltas = []
        stats = {
            "flops": 0.0, "memory": 0.0, "memory_saved": 0.0, 
            "communication": 0.0, "sparsity": 0.0,
            "skipped": 0, 
            "nonzero": 0, "total": 0,
            "effective_compute_ratio": 0.0,
            "avg_loss_change": 0.0,
            "loss_threshold": 0.0,
            "skipped_ratio_sum": 0.0,
            "layer_flops": {},
            "layer_communication": {}
        }

        # Determine method type
        method = "FedTinyProp"
        if use_dense_baseline:
            method = "Dense"
        elif use_fedprox:
            method = "FedProx"
        elif use_fedprune:
            method = "FedPrune"
        elif use_rigl:
            method = "RigL"

        # Set current round for all clients
        for client in clients:
            client.model.current_round = rnd
            # Only adjust TinyProp parameters if not using dense baseline or RigL
            if not use_dense_baseline and not use_rigl:
                client.model.tpLayer.adjust_loss_threshold(rnd, rounds)
                client.model.tpLayer.reset_batch_stats()

        # Client training phase
        for client_idx, client in enumerate(clients):
            print(f"\n[Training Debug] Training client {client_idx}")
            try:
                parameters, num_examples, metrics = client.fit(
                    [val.cpu().numpy() for val in global_params.values()],
                    config={"local_epochs": local_epochs}
                )
                client_deltas.append(client.weight_deltas)
            except Exception as e:
                print(f"[Training Debug][Client {client_idx}] Error during training: {str(e)}")
                continue
        
        # Server aggregation phase
        if client_deltas:
            global_model, agg_stats = aggregator_fn(
                client_deltas,
                global_model,
                model_name,
                tinyprop_params,
                dataset_sizes=dataset_sizes
            )

        # Evaluate global model
        acc = sum(client.local_evaluate(client.test_loader) for client in clients) / len(clients)
        
        # Update metrics log with consistent communication metrics
        metrics_log["timestamp"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        metrics_log["round"].append(rnd + 1)
        metrics_log["accuracy"].append(acc)
        metrics_log["flops"].append(sum(client.last_flops for client in clients) / len(clients))
        metrics_log["memory"].append(max(client.last_mem for client in clients))
        metrics_log["memory_saved"].append(sum(client.last_mem_saved for client in clients) / len(clients))
        
        # Get communication metrics from aggregator stats
        if isinstance(agg_stats, dict):
            total_comm = agg_stats["communication_bytes"]  # Total = download + upload
            download_bytes = agg_stats["download_bytes"]   # Total download for all clients
            upload_bytes = agg_stats["upload_bytes"]       # Total upload from all clients
            model_size = agg_stats["model_size_bytes"]     # Size of model
        else:
            # Fallback to client stats if aggregator stats not available
            total_comm = sum(client.last_comm for client in clients)
            download_bytes = sum(client.communication_metrics['download_bytes'] for client in clients)
            upload_bytes = sum(client.communication_metrics['upload_bytes'] for client in clients)
            model_size = clients[0].communication_metrics['model_size_bytes']
        
        metrics_log["communication"].append(total_comm)
        metrics_log["download_bytes"].append(download_bytes)
        metrics_log["upload_bytes"].append(upload_bytes)
        metrics_log["model_size_bytes"].append(model_size)
        
        # Calculate other metrics
        total_batches = sum(len(client.train_loader) for client in clients)
        total_skipped = sum(client.num_skipped_batches for client in clients)
        effective_compute_ratio = 1.0 - (total_skipped / total_batches) if total_batches > 0 else 0.0
        
        metrics_log["sparsity"].append(sum(client.last_sparsity for client in clients) / len(clients))
        metrics_log["skipped_batches"].append(total_skipped)
        metrics_log["effective_compute_ratio"].append(effective_compute_ratio)
        metrics_log["compression_ratio"].append(model_size / upload_bytes if upload_bytes > 0 else 1.0)
        metrics_log["client_eval_history"].append(
            {cid: client.local_evaluate(client.test_loader) for cid, client in enumerate(clients)}
        )
        
        # Add RigL-specific metrics if enabled
        if use_rigl:
            rigl_sparsity = sum(client.rigl_metrics["current_sparsity"] for client in clients) / len(clients) if clients else 0.0
            metrics_log["rigl_sparsity"].append(rigl_sparsity)
            
            # Log RigL mask updates (only for logging, not in metrics_log)
            total_mask_updates = sum(client.rigl_metrics["mask_updates"] for client in clients)
            print(f"[Training Debug] RigL mask updates: {total_mask_updates}")
        else:
            # Add a placeholder value for non-RigL runs to keep array lengths consistent
            metrics_log["rigl_sparsity"].append(0.0)

        # Save to single CSV file
        if csv_log_path:
            try:
                # Create DataFrame from current metrics
                df = pd.DataFrame({k: v for k, v in metrics_log.items() if len(v) > 0})
                
                # Verify all columns have the same length
                lengths = {k: len(v) for k, v in df.items()}
                if len(set(lengths.values())) != 1:
                    print(f"[Warning] Inconsistent array lengths: {lengths}")
                    # Find the minimum length
                    min_length = min(lengths.values())
                    # Truncate all arrays to the minimum length
                    df = df.iloc[:min_length]
                
                df.to_csv(csv_log_path, index=False)
                print(f"\n[INFO] Updated metrics saved to {csv_log_path}")
            except Exception as e:
                print(f"[Error] Failed to save metrics: {str(e)}")
                print(f"Metrics lengths: {lengths}")

        if early_stopper and early_stopper.step(acc, rnd):
            print(f"\n[Early Stop] Triggered at round {rnd+1}!")
            break

        history["train_loss"].append(0.0)
        history["train_accuracy"].append(acc)
        history["test_accuracy"].append(acc)
        history["sparsity"].append(metrics_log["sparsity"][-1])
        history["energy"].append(0.0)
        history["communication"].append(total_comm)

        # Calculate quantization metrics
        round_quantization_errors = []
        round_scale_factors = []
        for client in clients:
            if hasattr(client, 'get_quantization_metrics'):
                error, scale = client.get_quantization_metrics()
                round_quantization_errors.append(error)
                round_scale_factors.append(scale)
        
        avg_quantization_error = np.mean(round_quantization_errors) if round_quantization_errors else 0.0
        avg_scale_factor = np.mean(round_scale_factors) if round_scale_factors else 1.0
        quantization_errors.append(avg_quantization_error)
        avg_scale_factors.append(avg_scale_factor)

        # Calculate effective compute ratio correctly
        total_batches = sum(len(client.train_loader) for client in clients)
        total_skipped = sum(client.num_skipped_batches for client in clients)
        effective_compute_ratio = 1.0 - (total_skipped / total_batches) if total_batches > 0 else 0.0

        # Calculate compression ratio correctly
        total_original_size = sum(client.communication_metrics['model_size_bytes'] for client in clients)
        total_compressed_size = sum(client.communication_metrics['upload_bytes'] for client in clients)
        compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0

        # Update consolidated metrics with current timestamp
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        consolidated_metrics["timestamp"].append(current_timestamp)
        consolidated_metrics["round"].append(rnd + 1)
        consolidated_metrics["accuracy"].append(acc)
        consolidated_metrics["flops"].append(sum(client.last_flops for client in clients) / len(clients))
        consolidated_metrics["memory"].append(max(client.last_mem for client in clients))
        consolidated_metrics["memory_saved"].append(sum(client.last_mem_saved for client in clients) / len(clients))
        consolidated_metrics["communication"].append(total_comm)
        consolidated_metrics["sparsity"].append(sum(client.last_sparsity for client in clients) / len(clients))
        consolidated_metrics["skipped_batches"].append(total_skipped)
        consolidated_metrics["effective_compute_ratio"].append(effective_compute_ratio)
        consolidated_metrics["compression_ratio"].append(compression_ratio)
        consolidated_metrics["download_bytes"].append(download_bytes)
        consolidated_metrics["upload_bytes"].append(upload_bytes)
        consolidated_metrics["model_size_bytes"].append(model_size)
        consolidated_metrics["quantization_error"].append(avg_quantization_error)
        consolidated_metrics["avg_scale_factor"].append(avg_scale_factor)
        consolidated_metrics["method"].append(method)

        # Save consolidated metrics to a single CSV file
        if save_dir and save_interval > 0 and rnd % save_interval == 0:
            metrics_df = pd.DataFrame(consolidated_metrics)
            # Ensure timestamp is the first column
            cols = metrics_df.columns.tolist()
            cols.remove('timestamp')
            cols.insert(0, 'timestamp')
            metrics_df = metrics_df[cols]
            metrics_df.to_csv(
                os.path.join(save_dir, f"{partition_type}_{model_name}_consolidated_metrics.csv"),
                index=False
            )

        # Append detailed metrics
        append_to_training_log_csv(
            filepath=os.path.join(save_dir, f"{partition_type}_{model_name}_detailed_logs.csv"),
            round_num=rnd + 1,
            accuracy=acc,
            flops=metrics_log["flops"][-1],
            memory_bytes=metrics_log["memory"][-1],
            memory_saved=metrics_log["memory_saved"][-1],
            communication_bytes=metrics_log["communication"][-1],
            sparsity=metrics_log["sparsity"][-1],
            skipped_batches=metrics_log["skipped_batches"][-1],
            effective_compute_ratio=metrics_log["effective_compute_ratio"][-1],
            compression_ratio=metrics_log["compression_ratio"][-1],
            quantization_error=avg_quantization_error,
            avg_scale_factor=avg_scale_factor
        )

        # After aggregation, collect all metrics for this round
        round_metrics = {
            'round': rnd + 1,
            'accuracy': float(acc),
            'flops': float(sum(client.last_flops for client in clients) / len(clients)),
            'memory': float(max(client.last_mem for client in clients)),
            'memory_saved': float(sum(client.last_mem_saved for client in clients) / len(clients)),
            'communication': float(total_comm),
            'sparsity': float(sum(client.last_sparsity for client in clients) / len(clients)),
            'skipped_batches': int(total_skipped),
            'effective_compute_ratio': float(effective_compute_ratio),
            'compression_ratio': float(compression_ratio),
            'download_bytes': float(download_bytes),
            'upload_bytes': float(upload_bytes),
            'model_size_bytes': float(model_size)
        }

        # Add human-readable metrics
        round_metrics['communication_KB'] = round_metrics['communication'] / 1024
        round_metrics['communication_MB'] = round_metrics['communication_KB'] / 1024
        round_metrics['download_KB'] = round_metrics['download_bytes'] / 1024
        round_metrics['upload_KB'] = round_metrics['upload_bytes'] / 1024
        round_metrics['model_size_KB'] = round_metrics['model_size_bytes'] / 1024

        # Append all metrics at once to ensure synchronization
        for key in metrics_log.keys():
            if key in round_metrics:
                metrics_log[key].append(round_metrics[key])

        # Save metrics to CSV after each round
        if csv_log_path:
            try:
                # Create DataFrame from current metrics
                df = pd.DataFrame({k: v for k, v in metrics_log.items() if len(v) > 0})
                
                # Verify all columns have the same length
                lengths = {k: len(v) for k, v in df.items()}
                if len(set(lengths.values())) != 1:
                    print(f"[Warning] Inconsistent array lengths: {lengths}")
                    # Find the minimum length
                    min_length = min(lengths.values())
                    # Truncate all arrays to the minimum length
                    df = df.iloc[:min_length]
                
                df.to_csv(csv_log_path, index=False)
                print(f"\n[INFO] Updated metrics saved to {csv_log_path}")
            except Exception as e:
                print(f"[Error] Failed to save metrics: {str(e)}")
                print(f"Metrics lengths: {lengths}")

        # Print round summary
        print(f"[Training Debug][Round {rnd+1}/{rounds}] Accuracy: {acc:.4f}, Sparsity: {metrics_log['sparsity'][-1]:.4f}")
        print(f"[Training Debug][Round {rnd+1}/{rounds}] FLOPs: {metrics_log['flops'][-1]:.2f}, Memory: {metrics_log['memory'][-1]:.2f} bytes")
        print(f"[Training Debug][Round {rnd+1}/{rounds}] Communication: {metrics_log['communication'][-1]:.2f} bytes")
        print(f"[Training Debug][Round {rnd+1}/{rounds}] Compression ratio: {metrics_log['compression_ratio'][-1]:.2f}")
        print(f"[Training Debug][Round {rnd+1}/{rounds}] Skipped batches: {metrics_log['skipped_batches'][-1]}, Effective compute ratio: {metrics_log['effective_compute_ratio'][-1]:.4f}")
        
        # Print RigL-specific metrics if enabled
        if use_rigl:
            print(f"[Training Debug][Round {rnd+1}/{rounds}] RigL sparsity: {metrics_log['rigl_sparsity'][-1]:.4f}")
            
        # Save intermediate model checkpoints if requested
        if save_dir and save_interval > 0 and (rnd + 1) % save_interval == 0:
            checkpoint_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Determine model type
            model_type = "dense" if use_dense_baseline else "tinyprop" if not use_rigl else "rigl"
            
            # Save intermediate checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_{model_name}_round{rnd+1}.pt")
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"[INFO] Saved intermediate {model_type} model checkpoint to {checkpoint_path}")

    # Save final model checkpoints for hardware evaluation
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create model checkpoint directory
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model checkpoint
        model_type = "dense" if use_dense_baseline else "tinyprop" if not use_rigl else "rigl"
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_{model_name}.pt")
        
        # Save model state dictionary
        torch.save(global_model.state_dict(), checkpoint_path)
        print(f"\n[INFO] Saved {model_type} model checkpoint to {checkpoint_path}")
        
        # Save full checkpoint with metrics
        full_checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_{model_name}_full.pt")
        torch.save({
            'global_model_state_dict': global_model.state_dict(),
            'rounds': rounds,
            'final_accuracy': consolidated_metrics["accuracy"],
            'sparsity': consolidated_metrics["sparsity"],
            'compression_ratio': consolidated_metrics["compression_ratio"],
            'model_name': model_name,
            'model_type': model_type,
            'use_dense_baseline': use_dense_baseline,
            'use_rigl': use_rigl,
            'use_fedprox': use_fedprox,
            'use_fedprune': use_fedprune
        }, full_checkpoint_path)
        print(f"[INFO] Saved full {model_type} checkpoint with metrics to {full_checkpoint_path}")
    
    return (
        global_model,
        consolidated_metrics["accuracy"],
        consolidated_metrics["flops"],
        consolidated_metrics["memory"],
        consolidated_metrics["memory_saved"],
        consolidated_metrics["communication"],
        consolidated_metrics["sparsity"],
        consolidated_metrics["skipped_batches"],
        consolidated_metrics["effective_compute_ratio"],
        metrics_log["client_eval_history"],
        consolidated_metrics["compression_ratio"],
        consolidated_metrics["quantization_error"],
        consolidated_metrics["avg_scale_factor"],
        history
    )
