from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from models.model import get_tinyprop_model 
from models.config import get_tinyprop_config, get_dense_config
from utils.early_stopping import EarlyStoppingMonitor
from utils.save_results import append_to_training_log_csv, save_training_logs_csv
from utils.adaptive_sparsification import AdaptiveSparsifier
from models.model import get_tinyprop_model
from clients.aggregators import sparse_fedavg_aggregate, standard_fedavg_aggregate
from utils.adaptive_sparsification import AdaptiveSparsifier
from models.tinyProp import get_phi_k
import numpy as np
import random
import os
import pandas as pd
from datetime import datetime

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
    use_dense_baseline: bool = False
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
        tinyprop_params = get_dense_config(model_name)["tinyprop_params"]
    else:
        config = get_tinyprop_config(model_name)

    global_model = get_tinyprop_model(model_name, tinyprop_params)

    if num_clients is not None and num_rounds is not None and partition_type is not None and alpha is not None:
        print(f"\n[Training Debug] Starting federated training with {num_clients} clients")
        print(f"[Training Debug] Partition type: {partition_type}, alpha: {alpha}")
        
        global_model = get_tinyprop_model(model_name, config["tinyprop_params"])
        global_params = global_model.state_dict()
        clients = []
        dataset_sizes = []
        
        print("\n[Training Debug] Initializing clients...")
        for i in range(num_clients):
            train_loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)
            test_loader = DataLoader(testset, batch_size=32, shuffle=False)
            dataset_sizes.append(len(train_loader.dataset))
            
            client = FederatedClient(
                model=get_tinyprop_model(model_name, tinyprop_params),
                train_loader=train_loader,
                test_loader=test_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dataset_name="fashionmnist"
            )
            clients.append(client)
            print(f"[Training Debug] Client {i} initialized with {dataset_sizes[i]} samples")
        
        client_deltas = []
        quantization_errors = []
        avg_scale_factors = []
        for round_num in range(num_rounds):
            print(f"\n[Training Debug] Starting round {round_num + 1}/{num_rounds}")
            
            for client_idx, client in enumerate(clients):
                print(f"\n[Training Debug] Training client {client_idx}")
                try:
                    parameters, num_examples, metrics = client.fit(
                        [val.cpu().numpy() for val in global_params.values()],
                        config={"local_epochs": local_epochs}
                    )
                    
                    print(f"\n[Training Debug] Client {client_idx} weight_deltas:")
                    print(f"Keys: {list(client.weight_deltas.keys())}")
                    for param_name, update in client.weight_deltas.items():
                        print(f"\n[Training Debug] Processing {param_name}")
                        print(f"Update type: {type(update)}")
                        print(f"Update content: {update}")
                        
                        if not (isinstance(update, tuple) and len(update) == 2):
                            print(f"[Training Debug][Client {client_idx}] Malformed update detected for param '{param_name}': {update}")
                        elif not (isinstance(update[0], torch.Tensor) and isinstance(update[1], torch.Tensor)):
                            print(f"[Training Debug][Client {client_idx}] Non-tensor update detected for param '{param_name}': {update}")
                        else:
                            indices, values = update
                            print(f"[Training Debug][Client {client_idx}] Param '{param_name}' update shapes: indices={indices.shape}, values={values.shape}")
                    
                    client_deltas.append(client.weight_deltas)
                    
                except Exception as e:
                    print(f"[Training Debug][Client {client_idx}] Error during training: {str(e)}")
                    continue
            
            print("\n[Server Debug] Starting aggregation of client updates...")
            total_updates = 0
            total_skipped = 0
            
            for client_idx, deltas in enumerate(client_deltas):
                client_weight = len(clients[client_idx].train_loader.dataset) / sum(len(c.train_loader.dataset) for c in clients)
                print(f"\n[Server Debug] Processing client {client_idx} (weight: {client_weight:.4f})")
                
                for param_name, (indices, values) in deltas.items():
                    if isinstance(indices, torch.Tensor) and isinstance(values, torch.Tensor):
                        total_updates += 1
                        param = global_model.state_dict()[param_name]
                        if param.device != values.device:
                            values = values.to(param.device)
                            indices = indices.to(param.device)
                        
                        param.view(-1)[indices] += values * client_weight
                    else:
                        total_skipped += 1
                        print(f"[Server Debug] Skipping malformed update for {param_name}")
            
            print("\n[Server Debug] Aggregation Statistics:")
            print(f"Total clients processed: {len(client_deltas)}")
            print(f"Parameters skipped: {total_skipped}")
            print(f"Parameters updated: {total_updates}/{total_updates + total_skipped}")
            
            total_comm = 0
            for deltas in client_deltas:
                for name, (indices, values) in deltas.items():
                    if isinstance(indices, torch.Tensor) and isinstance(values, torch.Tensor):
                        total_comm += indices.numel() + values.numel()
            print(f"Total communication: {total_comm * 4 / 1024:.2f}KB")  
            
            global_model.load_state_dict(global_params)
            
            acc = sum(client.local_evaluate(client.test_loader) for client in clients) / len(clients)
            print(f"\n[Server Debug] Global model accuracy: {acc:.4f}")
            
            metrics_log["accuracy"].append(acc)
            metrics_log["flops"].append(sum(client.last_flops for client in clients) / len(clients))
            metrics_log["memory"].append(max(client.last_mem for client in clients))
            metrics_log["memory_saved"].append(sum(client.last_mem_saved for client in clients) / len(clients))
            metrics_log["communication"].append(sum(client.last_comm for client in clients) / len(clients))
            metrics_log["sparsity"].append(sum(client.last_sparsity for client in clients) / len(clients))
            metrics_log["skipped_batches"].append(sum(client.num_skipped_batches for client in clients))
            metrics_log["effective_compute_ratio"].append(1 - sum(client.num_skipped_batches for client in clients) / sum(len(client.train_loader) for client in clients))
            metrics_log["compression_ratio"].append(sum(client.compression_ratio for client in clients) / len(clients))
            
            
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
            
            client_deltas = []
        
            # Save metrics
            if save_dir and save_interval > 0 and round_num % save_interval == 0:
                # Save training logs after metrics_log["communication"] is updated
                save_training_logs_csv(
                    os.path.join(save_dir, f"{partition_type}_{model_name}_training_logs.csv"),
                    metrics_log["accuracy"],
                    metrics_log["flops"],
                    metrics_log["memory"],
                    metrics_log["communication"],
                    metrics_log["sparsity"],
                    quantization_errors,
                    memory_saved=metrics_log["memory_saved"]
                )

        return (
            global_model,
            metrics_log["accuracy"],
            metrics_log["flops"],
            metrics_log["memory"],
            metrics_log["memory_saved"],
            metrics_log["communication"],
            metrics_log["sparsity"],
            metrics_log["skipped_batches"],
            metrics_log["effective_compute_ratio"],
            metrics_log["client_eval_history"],
            metrics_log["compression_ratio"],
            metrics_log["quantization_errors"],
            metrics_log["avg_scale_factors"]
        )

    global_model = get_tinyprop_model(model_name, tinyprop_params).to(device)

    clients = []
    for i, dataset in enumerate(client_datasets):
        client_model = get_tinyprop_model(model_name, tinyprop_params).to(device)
        client_model.load_state_dict(global_model.state_dict())
        clients.append(FederatedClient(
            client_id=i,
            model=client_model,
            train_loader=DataLoader(dataset, batch_size=32, shuffle=True),
            test_loader=DataLoader(testset, batch_size=32, shuffle=False),
            cfg=config,
            device=device,
            dataset_name=model_name
        ))

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
        "avg_scale_factor": []
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

        # Set current round for all clients
        for client in clients:
            client.model.current_round = rnd
            if not use_dense_baseline:
                client.model.tpLayer.adjust_loss_threshold(rnd, rounds)
                client.model.tpLayer.reset_batch_stats()

        # Client training phase
        for client_idx, client in enumerate(clients):
            parameters, num_examples, metrics = client.fit(
                [val.cpu().numpy() for val in global_params.values()],
                config={"local_epochs": local_epochs}
            )
            
            client_deltas.append(client.weight_deltas)
            
            # Update statistics from client metrics
            stats["flops"] += client.last_flops
            stats["memory"] = max(stats["memory"], client.last_mem)
            stats["memory_saved"] += client.last_mem_saved
            stats["communication"] += client.last_comm
            stats["sparsity"] += client.last_sparsity
            stats["skipped"] += client.num_skipped_batches
            
            # Track layer-wise metrics
            client_metrics = client.compute_metrics()
            for layer_name, layer_flops in client_metrics.get("layer_flops", {}).items():
                if layer_name not in stats["layer_flops"]:
                    stats["layer_flops"][layer_name] = 0
                stats["layer_flops"][layer_name] += layer_flops
            
            for layer_name, layer_comm in client_metrics.get("layer_communication", {}).items():
                if layer_name not in stats["layer_communication"]:
                    stats["layer_communication"][layer_name] = 0
                stats["layer_communication"][layer_name] += layer_comm

        # Average the statistics across clients
        num_clients = len(clients)
        if num_clients > 0:
            for key in ["flops", "memory_saved", "communication", "sparsity", 
                       "effective_compute_ratio", "avg_loss_change", "loss_threshold", 
                       "skipped_ratio_sum"]:
                stats[key] /= num_clients

        # Server aggregation phase
        global_model, agg_stats = aggregator_fn(
            client_deltas, 
            global_model, 
            model_name, 
            tinyprop_params, 
            **{**aggregator_kwargs, "dataset_sizes": [len(c.train_loader.dataset) for c in clients]}
        )

        # Evaluate global model
        acc = sum(client.local_evaluate(client.test_loader) for client in clients) / len(clients)
        
        # Update metrics log with consistent communication metrics
        metrics_log["timestamp"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        metrics_log["round"].append(rnd + 1)
        metrics_log["accuracy"].append(acc)
        metrics_log["flops"].append(stats["flops"])
        metrics_log["memory"].append(stats["memory"])
        metrics_log["memory_saved"].append(stats["memory_saved"])
        
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
        
        metrics_log["sparsity"].append(stats["sparsity"])
        metrics_log["skipped_batches"].append(total_skipped)
        metrics_log["effective_compute_ratio"].append(effective_compute_ratio)
        metrics_log["compression_ratio"].append(model_size / upload_bytes if upload_bytes > 0 else 1.0)
        metrics_log["client_eval_history"].append(
            {cid: client.local_evaluate(client.test_loader) for cid, client in enumerate(clients)}
        )

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
        history["communication"].append(stats["communication"])

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
        consolidated_metrics["flops"].append(stats["flops"])
        consolidated_metrics["memory"].append(stats["memory"])
        consolidated_metrics["memory_saved"].append(stats["memory_saved"])
        consolidated_metrics["communication"].append(stats["communication"])
        consolidated_metrics["sparsity"].append(stats["sparsity"])
        consolidated_metrics["skipped_batches"].append(stats["skipped"])
        consolidated_metrics["effective_compute_ratio"].append(effective_compute_ratio)
        consolidated_metrics["compression_ratio"].append(compression_ratio)
        consolidated_metrics["download_bytes"].append(sum(client.communication_metrics['download_bytes'] for client in clients) / len(clients))
        consolidated_metrics["upload_bytes"].append(sum(client.communication_metrics['upload_bytes'] for client in clients) / len(clients))
        consolidated_metrics["model_size_bytes"].append(sum(client.communication_metrics['model_size_bytes'] for client in clients) / len(clients))
        consolidated_metrics["quantization_error"].append(avg_quantization_error)
        consolidated_metrics["avg_scale_factor"].append(avg_scale_factor)

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
            'flops': float(stats["flops"]),
            'memory': float(stats["memory"]),
            'memory_saved': float(stats["memory_saved"]),
            'communication': float(total_comm),
            'sparsity': float(stats["sparsity"]),
            'skipped_batches': int(total_skipped),
            'effective_compute_ratio': float(effective_compute_ratio),
            'compression_ratio': float(compression_ratio),
            'download_bytes': float(model_size),
            'upload_bytes': float(total_comm),
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
        print(f"\nRound {rnd + 1} Summary:")
        print(f"Accuracy: {round_metrics['accuracy']:.4f}")
        print(f"Communication: {round_metrics['communication_MB']:.2f}MB (Download: {round_metrics['download_KB']:.2f}KB, Upload: {round_metrics['upload_KB']:.2f}KB)")
        print(f"Model Size: {round_metrics['model_size_KB']:.2f}KB")
        print(f"Compression Ratio: {round_metrics['compression_ratio']:.2f}x")
        print(f"Sparsity: {round_metrics['sparsity']:.2%}")
        print(f"Skipped Batches: {round_metrics['skipped_batches']}")
        print(f"Effective Compute Ratio: {round_metrics['effective_compute_ratio']:.4f}")

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
