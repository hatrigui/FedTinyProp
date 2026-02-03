from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from clients.rigl_client import FederatedRigLClient
from models.model import get_tinyprop_model 
from models.config import get_tinyprop_config, get_dense_config
from utils.early_stopping import EarlyStoppingMonitor
from utils.save_results import append_to_training_log_csv, save_training_logs_csv, save_per_client_metrics_csv
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
    track_per_client_metrics: bool = False,
    enable_profiling: bool = False,
    ablation: str = "full",
    fixed_phi: float = 1.0,
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
        "method": [],
        "accuracy": [],
        "flops": [],
        "dense_flops_ref": [],
        "saved_flops_est": [],
        "saved_flops_ratio": [],
        "rigl_sparsity": [],  # Always initialize as empty list
        "memory": [],
        "memory_saved": [],
        "sram_usage": [],
        "latency_ms": [],
        "time_forward_ms": [],
        "time_backward_ms": [],
        "time_grad_norm_ms": [],
        "time_controller_ms": [],
        "time_topk_ms": [],
        "time_quant_ms": [],
        "time_opt_step_ms": [],
        "time_delta_ms": [],
        "time_total_ms": [],
        "controller_overhead_pct": [],
        "topk_overhead_pct": [],
        "non_backprop_overhead_pct": [],
        "communication": [],
        "sparsity": [],
        "skipped_batches": [],
        "effective_compute_ratio": [],
        "compression_ratio": [],
        "download_bytes": [],
        "upload_bytes": [],
        "model_size_bytes": [],
        "dense_upload_bytes_ref": [],
        "saved_upload_bytes": [],
        "saved_upload_ratio": [],
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
        client_cfg["enable_profiling"] = enable_profiling

        ablation_l = (ablation or "full").lower()
        if not use_dense_baseline and not use_fedprune and not use_rigl:
            if ablation_l in ["full", "default"]:
                client_cfg["enable_controller"] = True
                client_cfg["enable_skip"] = True
                client_cfg["enable_topk"] = True
                client_cfg["adaptive_sparsity"] = True
                client_cfg["enable_zeta_tuning"] = True
            elif ablation_l in ["adapt_only", "adapt-only", "adapt"]:
                client_cfg["enable_controller"] = False
                client_cfg["enable_skip"] = False
                client_cfg["enable_topk"] = True
                client_cfg["adaptive_sparsity"] = True
                client_cfg["enable_zeta_tuning"] = True
            elif ablation_l in ["skip_only", "skip-only", "skip"]:
                client_cfg["enable_controller"] = True
                client_cfg["enable_skip"] = True
                client_cfg["enable_topk"] = False
                client_cfg["adaptive_sparsity"] = False
                client_cfg["enable_zeta_tuning"] = True
            elif ablation_l in ["fixed_topk", "fixed-topk", "fixed"]:
                client_cfg["enable_controller"] = False
                client_cfg["enable_skip"] = False
                client_cfg["enable_topk"] = True
                client_cfg["adaptive_sparsity"] = False
                client_cfg["fixed_phi"] = float(fixed_phi)
                client_cfg["enable_zeta_tuning"] = False
        
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
        "dense_flops_ref": [],
        "saved_flops_est": [],
        "saved_flops_ratio": [],
        "memory": [],
        "memory_saved": [],
        "sram_usage": [],
        "latency_ms": [],
        "time_forward_ms": [],
        "time_backward_ms": [],
        "time_grad_norm_ms": [],
        "time_controller_ms": [],
        "time_topk_ms": [],
        "time_quant_ms": [],
        "time_opt_step_ms": [],
        "time_delta_ms": [],
        "time_total_ms": [],
        "controller_overhead_pct": [],
        "topk_overhead_pct": [],
        "non_backprop_overhead_pct": [],
        "communication": [],
        "sparsity": [],
        "skipped_batches": [],
        "effective_compute_ratio": [],
        "compression_ratio": [],
        "download_bytes": [],
        "upload_bytes": [],
        "model_size_bytes": [],
        "dense_upload_bytes_ref": [],
        "saved_upload_bytes": [],
        "saved_upload_ratio": [],
        "quantization_error": [],
        "avg_scale_factor": [],
        "smoothed_phi": [],  # Added smoothed gradient-norm signal metric
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
        else:
            ablation_l = (ablation or "full").lower()
            if ablation_l not in ["full", "default"]:
                method = f"FedTinyProp_{ablation_l}"

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
        metrics_log["method"].append(method)
        metrics_log["accuracy"].append(acc)
        metrics_log["flops"].append(sum(client.last_flops for client in clients) / len(clients))
        metrics_log["memory"].append(max(client.last_mem for client in clients))
        metrics_log["memory_saved"].append(sum(client.last_mem_saved for client in clients) / len(clients))

        # Aggregate timing metrics (ms) if profiling is enabled; otherwise keep zeros.
        if enable_profiling:
            client_timing = [c.get_metrics() for c in clients]
            def _avg(key: str) -> float:
                vals = [m.get(key, 0.0) for m in client_timing]
                return float(sum(vals) / len(vals)) if vals else 0.0

            metrics_log["time_forward_ms"].append(_avg("time_forward_ms"))
            metrics_log["time_backward_ms"].append(_avg("time_backward_ms"))
            metrics_log["time_grad_norm_ms"].append(_avg("time_grad_norm_ms"))
            metrics_log["time_controller_ms"].append(_avg("time_controller_ms"))
            metrics_log["time_topk_ms"].append(_avg("time_topk_ms"))
            metrics_log["time_quant_ms"].append(_avg("time_quant_ms"))
            metrics_log["time_opt_step_ms"].append(_avg("time_opt_step_ms"))
            metrics_log["time_delta_ms"].append(_avg("time_delta_ms"))
            metrics_log["time_total_ms"].append(_avg("time_total_ms"))
        else:
            metrics_log["time_forward_ms"].append(0.0)
            metrics_log["time_backward_ms"].append(0.0)
            metrics_log["time_grad_norm_ms"].append(0.0)
            metrics_log["time_controller_ms"].append(0.0)
            metrics_log["time_topk_ms"].append(0.0)
            metrics_log["time_quant_ms"].append(0.0)
            metrics_log["time_opt_step_ms"].append(0.0)
            metrics_log["time_delta_ms"].append(0.0)
            metrics_log["time_total_ms"].append(0.0)
        
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
        if isinstance(agg_stats, dict) and "compression_ratio" in agg_stats:
            metrics_log["compression_ratio"].append(float(agg_stats["compression_ratio"]))
        else:
            dense_upload_total = model_size * len(clients)
            metrics_log["compression_ratio"].append(dense_upload_total / upload_bytes if upload_bytes > 0 else 1.0)

        # Overhead-vs-savings derived metrics (per round)
        dense_flops_ref = (
            float(sum(getattr(c, "last_dense_flops_ref", 0.0) for c in clients) / len(clients))
            if clients else 0.0
        )
        actual_flops = float(metrics_log["flops"][-1]) if metrics_log["flops"] else 0.0
        saved_flops_est = max(0.0, dense_flops_ref - actual_flops)
        saved_flops_ratio = (saved_flops_est / dense_flops_ref) if dense_flops_ref > 0 else 0.0

        metrics_log["dense_flops_ref"].append(dense_flops_ref)
        metrics_log["saved_flops_est"].append(saved_flops_est)
        metrics_log["saved_flops_ratio"].append(saved_flops_ratio)

        # Communication savings vs dense uplink (dense upload ~ model_size per client)
        dense_upload_bytes_ref = float(model_size) * float(len(clients)) if clients else 0.0
        saved_upload_bytes = max(0.0, float(dense_upload_bytes_ref) - float(upload_bytes))
        saved_upload_ratio = (saved_upload_bytes / dense_upload_bytes_ref) if dense_upload_bytes_ref > 0 else 0.0

        metrics_log["dense_upload_bytes_ref"].append(dense_upload_bytes_ref)
        metrics_log["saved_upload_bytes"].append(saved_upload_bytes)
        metrics_log["saved_upload_ratio"].append(saved_upload_ratio)

        # Timing overhead percentages (meaningful only when enable_profiling=True)
        t_total = float(metrics_log["time_total_ms"][-1]) if metrics_log["time_total_ms"] else 0.0
        t_controller = float(metrics_log["time_controller_ms"][-1]) if metrics_log["time_controller_ms"] else 0.0
        t_topk = float(metrics_log["time_topk_ms"][-1]) if metrics_log["time_topk_ms"] else 0.0

        t_non_backprop = float(
            (metrics_log["time_grad_norm_ms"][-1] if metrics_log["time_grad_norm_ms"] else 0.0)
            + (metrics_log["time_controller_ms"][-1] if metrics_log["time_controller_ms"] else 0.0)
            + (metrics_log["time_topk_ms"][-1] if metrics_log["time_topk_ms"] else 0.0)
            + (metrics_log["time_quant_ms"][-1] if metrics_log["time_quant_ms"] else 0.0)
            + (metrics_log["time_delta_ms"][-1] if metrics_log["time_delta_ms"] else 0.0)
        )

        metrics_log["controller_overhead_pct"].append((100.0 * t_controller / t_total) if t_total > 0 else 0.0)
        metrics_log["topk_overhead_pct"].append((100.0 * t_topk / t_total) if t_total > 0 else 0.0)
        metrics_log["non_backprop_overhead_pct"].append((100.0 * t_non_backprop / t_total) if t_total > 0 else 0.0)
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
        
        # Calculate average smoothed_phi across all clients
        if clients:
            smoothed_phi_vals = []
            for client in clients:
                v = getattr(client, 'smoothed_phi', 0.0)
                if v is None:
                    v = 0.0
                smoothed_phi_vals.append(float(v))
            avg_smoothed_phi = sum(smoothed_phi_vals) / len(smoothed_phi_vals)
        else:
            avg_smoothed_phi = 0.0

        # Update consolidated metrics with current timestamp
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        consolidated_metrics["timestamp"].append(current_timestamp)
        consolidated_metrics["round"].append(rnd + 1)
        consolidated_metrics["method"].append(method)
        consolidated_metrics["accuracy"].append(acc)
        consolidated_metrics["flops"].append(sum(client.last_flops for client in clients) / len(clients))
        consolidated_metrics["memory"].append(max(client.last_mem for client in clients))
        consolidated_metrics["memory_saved"].append(sum(client.last_mem_saved for client in clients) / len(clients))
        consolidated_metrics["smoothed_phi"].append(avg_smoothed_phi)  # Add smoothed gradient-norm signal

        # Mirror the timing breakdown to consolidated_metrics for convenience
        consolidated_metrics["time_forward_ms"].append(metrics_log["time_forward_ms"][-1])
        consolidated_metrics["time_backward_ms"].append(metrics_log["time_backward_ms"][-1])
        consolidated_metrics["time_grad_norm_ms"].append(metrics_log["time_grad_norm_ms"][-1])
        consolidated_metrics["time_controller_ms"].append(metrics_log["time_controller_ms"][-1])
        consolidated_metrics["time_topk_ms"].append(metrics_log["time_topk_ms"][-1])
        consolidated_metrics["time_quant_ms"].append(metrics_log["time_quant_ms"][-1])
        consolidated_metrics["time_opt_step_ms"].append(metrics_log["time_opt_step_ms"][-1])
        consolidated_metrics["time_delta_ms"].append(metrics_log["time_delta_ms"][-1])
        consolidated_metrics["time_total_ms"].append(metrics_log["time_total_ms"][-1])

        consolidated_metrics["dense_flops_ref"].append(metrics_log["dense_flops_ref"][-1])
        consolidated_metrics["saved_flops_est"].append(metrics_log["saved_flops_est"][-1])
        consolidated_metrics["saved_flops_ratio"].append(metrics_log["saved_flops_ratio"][-1])

        consolidated_metrics["dense_upload_bytes_ref"].append(metrics_log["dense_upload_bytes_ref"][-1])
        consolidated_metrics["saved_upload_bytes"].append(metrics_log["saved_upload_bytes"][-1])
        consolidated_metrics["saved_upload_ratio"].append(metrics_log["saved_upload_ratio"][-1])

        consolidated_metrics["controller_overhead_pct"].append(metrics_log["controller_overhead_pct"][-1])
        consolidated_metrics["topk_overhead_pct"].append(metrics_log["topk_overhead_pct"][-1])
        consolidated_metrics["non_backprop_overhead_pct"].append(metrics_log["non_backprop_overhead_pct"][-1])
        
        # Add SRAM and latency metrics
        try:
            # Get current SRAM usage
            from utils.rpi_utils import get_raspberry_pi_memory_usage
            memory_info = get_raspberry_pi_memory_usage()
            sram_usage = memory_info.get('sram_used_mb', 0)
            
            # Measure inference latency
            import time
            
            # Create a small dummy batch for latency measurement
            dummy_input = torch.rand(4, 3, 32, 32).to(device)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                global_model(dummy_input)
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            # Get CPU temperature
            from utils.rpi_utils import get_raspberry_pi_temperature
            temperature = get_raspberry_pi_temperature()
            
            # Get CPU usage
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage details
            mem = psutil.virtual_memory()
            total_memory_mb = mem.total / (1024 * 1024)
            available_memory_mb = mem.available / (1024 * 1024)
            used_memory_mb = mem.used / (1024 * 1024)
            memory_percent = mem.percent
            
            consolidated_metrics["sram_usage"].append(sram_usage)
            consolidated_metrics["latency_ms"].append(latency)
            
            # Also update the metrics_log
            if len(metrics_log["sram_usage"]) <= rnd:
                metrics_log["sram_usage"].append(sram_usage)
            if len(metrics_log["latency_ms"]) <= rnd:
                metrics_log["latency_ms"].append(latency)
                
            # Create embedded metrics dictionary for this round
            embedded_metrics = {
                "timestamp": current_timestamp,
                "round": rnd + 1,
                "sram_usage_mb": sram_usage,
                "latency_ms": latency,
                "cpu_temperature": temperature,
                "cpu_percent": cpu_percent,
                "total_memory_mb": total_memory_mb,
                "available_memory_mb": available_memory_mb,
                "used_memory_mb": used_memory_mb,
                "memory_percent": memory_percent,
                "flops": sum(client.last_flops for client in clients) / len(clients),
                "memory": max(client.last_mem for client in clients),
                "memory_saved": sum(client.last_mem_saved for client in clients) / len(clients),
                "sparsity": sum(client.last_sparsity for client in clients) / len(clients),
                "effective_compute_ratio": effective_compute_ratio,
                "compression_ratio": compression_ratio,
                "model_size_bytes": model_size,
                "method": method
            }
            
            # Save embedded metrics to a separate CSV file
            if save_dir:
                if track_per_client_metrics:
                    # When tracking per-client metrics, we'll save embedded metrics with client info
                    embedded_metrics_csv_path = os.path.join(save_dir, f"{partition_type}_{model_name}_embedded_metrics.csv")
                    
                    # Collect metrics for each client
                    all_embedded_metrics = []
                    
                    for client_idx, client in enumerate(clients):
                        # Create a row for this client
                        client_metrics = {
                            "timestamp": current_timestamp,
                            "round": rnd + 1,
                            "client": f"client_{client_idx}",
                            "sram_usage_mb": sram_usage,
                            "latency_ms": latency,
                            "cpu_temperature": temperature,
                            "cpu_percent": cpu_percent,
                            "total_memory_mb": total_memory_mb,
                            "available_memory_mb": available_memory_mb,
                            "used_memory_mb": used_memory_mb,
                            "memory_percent": memory_percent,
                            "method": method
                        }
                        
                        # Add client-specific metrics
                        client_metrics.update(client.get_metrics())
                        all_embedded_metrics.append(client_metrics)
                    
                    # Add aggregated metrics row
                    embedded_metrics["client"] = "agg"  # Special identifier for aggregated metrics
                    all_embedded_metrics.append(embedded_metrics)
                    
                    # Convert to DataFrame
                    embedded_metrics_df = pd.DataFrame(all_embedded_metrics)
                else:
                    # Traditional format with one row per round (aggregated metrics only)
                    embedded_metrics_df = pd.DataFrame([embedded_metrics])
                    embedded_metrics_csv_path = os.path.join(save_dir, f"{partition_type}_{model_name}_embedded_metrics.csv")
                
                # Check if file exists to determine if we need to write headers
                file_exists = os.path.isfile(embedded_metrics_csv_path)
                
                # Append to the CSV file (or create it if it doesn't exist)
                embedded_metrics_df.to_csv(
                    embedded_metrics_csv_path,
                    mode='a' if file_exists else 'w',
                    header=not file_exists,
                    index=False
                )
                
        except Exception as e:
            print(f"[Warning] Failed to collect embedded metrics: {str(e)}")
            consolidated_metrics["sram_usage"].append(0)
            consolidated_metrics["latency_ms"].append(0)
            
            # Also update the metrics_log with zeros
            if len(metrics_log["sram_usage"]) <= rnd:
                metrics_log["sram_usage"].append(0)
            if len(metrics_log["latency_ms"]) <= rnd:
                metrics_log["latency_ms"].append(0)
        
        consolidated_metrics["communication"].append(total_comm)
        consolidated_metrics["sparsity"].append(sum(client.last_sparsity for client in clients) / len(clients))
        consolidated_metrics["skipped_batches"].append(total_skipped)
        consolidated_metrics["effective_compute_ratio"].append(effective_compute_ratio)
        consolidated_metrics["compression_ratio"].append(compression_ratio)
        consolidated_metrics["download_bytes"].append(download_bytes)
        consolidated_metrics["upload_bytes"].append(upload_bytes)
        consolidated_metrics["model_size_bytes"].append(model_size)
        
        # Append detailed metrics
        if not track_per_client_metrics:
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

        # Save metrics to CSV after each round
        if csv_log_path:
            try:
                if track_per_client_metrics:
                    # When tracking per-client metrics, we'll save per-client metrics directly to the main CSV file
                    
                    # Create aggregated metrics dictionary for this round
                    agg_metrics = {
                        "accuracy": acc,
                        "flops": sum(client.last_flops for client in clients) / len(clients),
                        "memory": max(client.last_mem for client in clients),
                        "memory_saved": sum(client.last_mem_saved for client in clients) / len(clients),
                        "communication": total_comm,
                        "sparsity": sum(client.last_sparsity for client in clients) / len(clients),
                        "skipped_batches": total_skipped,
                        "effective_compute_ratio": effective_compute_ratio,
                        "compression_ratio": compression_ratio,
                        "download_bytes": download_bytes,
                        "upload_bytes": upload_bytes,
                        "model_size_bytes": model_size
                    }
                    
                    # Create a new CSV file with the correct structure for per-client metrics
                    try:
                        # Create a new file path with a timestamp to avoid conflicts
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_dir = os.path.dirname(csv_log_path)
                        csv_basename = os.path.basename(csv_log_path)
                        csv_name, csv_ext = os.path.splitext(csv_basename)
                        
                        # Use the original path if it doesn't exist yet, otherwise create a new file
                        if not os.path.exists(csv_log_path):
                            new_csv_path = csv_log_path
                        else:
                            # Check if the file has the correct structure
                            try:
                                existing_df = pd.read_csv(csv_log_path, nrows=1)
                                required_columns = [
                                    "timestamp",
                                    "round",
                                    "client_id",
                                ] + [k for k in metrics_log.keys() if k not in ("timestamp", "round")]

                                if (
                                    "client_id" in existing_df.columns
                                    and all(col in existing_df.columns for col in required_columns)
                                ):
                                    # File has correct structure, use it
                                    new_csv_path = csv_log_path
                                else:
                                    # File has incorrect structure, create a new one
                                    new_csv_path = os.path.join(csv_dir, f"{csv_name}_with_client_id{csv_ext}")
                                    print(f"[INFO] Creating new CSV file with correct structure: {new_csv_path}")
                            except Exception:
                                # Can't read the file, create a new one
                                new_csv_path = os.path.join(csv_dir, f"{csv_name}_with_client_id{csv_ext}")
                                print(f"[INFO] Creating new CSV file with correct structure: {new_csv_path}")
                        
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)
                        
                        # Build rows for all clients + one aggregated row.
                        # Each client row includes its client-specific metrics, plus the same round-level
                        # aggregated/derived metrics (latency, overhead %, dense refs/savings, etc.)
                        # so this CSV can be analyzed without joining against the main per-round CSV.
                        base_round_metrics = {k: metrics_log[k][-1] for k in metrics_log.keys() if metrics_log.get(k)}
                        base_round_metrics.pop("timestamp", None)
                        base_round_metrics.pop("round", None)

                        all_metrics = []
                        timestamp = metrics_log["timestamp"][-1] if metrics_log.get("timestamp") else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        round_no = metrics_log["round"][-1] if metrics_log.get("round") else (rnd + 1)

                        # Per-client rows
                        for client_idx, client in enumerate(clients):
                            client_metrics = client.get_metrics()
                            metrics_row = {
                                "timestamp": timestamp,
                                "round": round_no,
                                "client_id": f"client_{client_idx}",
                                **base_round_metrics,
                            }

                            # Override round-level metrics with client-specific values
                            metrics_row["accuracy"] = client.local_evaluate(client.test_loader)
                            metrics_row["flops"] = client_metrics.get("flops", 0.0)
                            metrics_row["memory"] = client_metrics.get("memory", 0.0)
                            metrics_row["memory_saved"] = client_metrics.get("memory_saved", 0.0)
                            metrics_row["communication"] = client_metrics.get("communication", 0.0)
                            metrics_row["sparsity"] = client_metrics.get("sparsity", 0.0)
                            metrics_row["skipped_batches"] = client_metrics.get("skipped_batches", 0)
                            metrics_row["download_bytes"] = client_metrics.get("download_bytes", 0.0)
                            metrics_row["upload_bytes"] = client_metrics.get("upload_bytes", 0.0)
                            metrics_row["model_size_bytes"] = client_metrics.get("model_size_bytes", 0.0)
                            metrics_row["compression_ratio"] = client_metrics.get("compression_ratio", 1.0)

                            # If profiling is enabled, prefer per-client timing (more informative than round average)
                            for k in [
                                "time_forward_ms",
                                "time_backward_ms",
                                "time_grad_norm_ms",
                                "time_controller_ms",
                                "time_topk_ms",
                                "time_quant_ms",
                                "time_opt_step_ms",
                                "time_delta_ms",
                                "time_total_ms",
                            ]:
                                if k in client_metrics and client_metrics[k] is not None:
                                    metrics_row[k] = client_metrics[k]

                            all_metrics.append(metrics_row)

                        # Aggregated row (keeps the round-level metrics values)
                        agg_row = {
                            "timestamp": timestamp,
                            "round": round_no,
                            "client_id": "agg",
                            **base_round_metrics,
                        }
                        all_metrics.append(agg_row)

                        # Convert to DataFrame with a stable column order: client_id + the main per-round columns
                        df = pd.DataFrame(all_metrics)

                        required_columns = [
                            "timestamp",
                            "round",
                            "client_id",
                        ] + [k for k in metrics_log.keys() if k not in ("timestamp", "round")]
                        df = df.reindex(columns=required_columns)
                        
                        # Check if file exists to determine if we need to write headers
                        file_exists = os.path.isfile(new_csv_path)
                        
                        # Write to CSV with explicit column ordering
                        df.to_csv(
                            new_csv_path,
                            mode='a' if file_exists else 'w',
                            header=not file_exists,
                            index=False
                        )
                        
                        # Update the csv_log_path to point to the new file
                        if new_csv_path != csv_log_path:
                            csv_log_path = new_csv_path
                        
                        print(f"\n[INFO] Updated per-client metrics saved to {new_csv_path}")
                    except Exception as e:
                        print(f"[Error] Failed to save per-client metrics: {str(e)}")
                        # Don't fall back to the utility function as it might have the same issue
                else:
                    # Traditional format with one row per round (aggregated metrics only)
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
        
        # Create checkpoint data dictionary
        checkpoint_data = {
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
            'use_fedprune': use_fedprune,
            'track_per_client_metrics': track_per_client_metrics
        }
        
        # If tracking per-client metrics, include client-specific information
        if track_per_client_metrics:
            # Add final client metrics
            client_metrics = []
            for client_idx, client in enumerate(clients):
                client_data = {
                    'client_id': client_idx,
                    'accuracy': client.local_evaluate(client.test_loader)
                }
                client_data.update(client.get_metrics())
                client_metrics.append(client_data)
            
            checkpoint_data['client_metrics'] = client_metrics
        
        torch.save(checkpoint_data, full_checkpoint_path)
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
