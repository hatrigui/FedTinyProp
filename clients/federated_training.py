from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from models.model import get_tinyprop_model 
from models.config import get_tinyprop_config
from utils.early_stopping import EarlyStoppingMonitor
from utils.save_results import append_to_training_log_csv, save_training_logs_csv

from utils.adaptive_sparsification import AdaptiveSparsifier
from models.model import get_tinyprop_model
from clients.aggregators import sparse_fedavg_aggregate
import numpy as np
import random
from models.tinyProp import get_phi_k
import os

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
    quantization_bits=8
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if aggregator_kwargs is None:
        aggregator_kwargs = {}

    if "dataset_sizes" not in aggregator_kwargs:
        aggregator_kwargs["dataset_sizes"] = [len(ds) for ds in client_datasets]

    metrics_log = {
        "accuracy": [], "flops": [], "memory": [], "communication": [], "sparsity": [],
        "avg_grad_norm": [], "phi_k": [], "skipped_batches": [],
        "effective_compute_ratio": [], "compression_ratio": [], "client_eval_history": [],
        "avg_phi_k": [], "avg_loss_change": [], "loss_threshold": [], "skipped_ratio": [],
        "quantization_errors": [], "avg_scale_factors": []
    }

    config = get_tinyprop_config(model_name)
    global_model = get_tinyprop_model(model_name, tinyprop_params)

    if num_clients is not None and num_rounds is not None and partition_type is not None and alpha is not None:
        print(f"\n[Training Debug] Starting federated training with {num_clients} clients")
        print(f"[Training Debug] Partition type: {partition_type}, alpha: {alpha}")
        
        global_model = get_tinyprop_model(model_name, tinyprop_params)
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
            metrics_log["communication"].append(sum(client.last_comm for client in clients) / len(clients))
            metrics_log["sparsity"].append(sum(client.last_sparsity for client in clients) / len(clients))
            metrics_log["avg_grad_norm"].append(sum(client.last_avg_grad_norm for client in clients) / len(clients))
            metrics_log["phi_k"].append(sum(client.last_phi for client in clients) / len(clients))
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
            save_training_logs_csv(
                os.path.join(save_dir, f"{partition_type}_{model_name}_training_logs.csv"),
                metrics_log["accuracy"],
                metrics_log["flops"],
                metrics_log["memory"],
                metrics_log["communication"],
                metrics_log["sparsity"],
                quantization_errors
            )

        return (
            global_model,
            metrics_log["accuracy"],
            metrics_log["flops"],
            metrics_log["memory"],
            metrics_log["communication"],
            metrics_log["sparsity"],
            metrics_log["avg_grad_norm"],
            metrics_log["phi_k"],
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

    early_stopper = EarlyStoppingMonitor(patience=early_stopping_patience, delta=early_stopping_delta)
    sparsifier = AdaptiveSparsifier(initial_sparsity, target_sparsity, rounds, energy_budget=energy_budget)
    history = {k: [] for k in ["train_loss", "train_accuracy", "test_accuracy", "sparsity", "energy", "communication"]}

    # Initialize quantization metrics
    quantization_errors = []
    avg_scale_factors = []

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_deltas = []
        stats = {
            "flops": 0.0, "memory": 0.0, "communication": 0.0, "sparsity": 0.0,
            "grad_norm": 0.0, "phi_k": 0.0, "skipped": 0, "nonzero": 0, "total": 0,
            "effective_compute_ratio": 0.0,
            "avg_phi_k": 0.0,
            "avg_loss_change": 0.0,
            "loss_threshold": 0.0,
            "skipped_ratio_sum": 0.0  # Track sum of per-client skipped ratios
        }

        # Set current round for all clients and adjust thresholds
        for client in clients:
            client.model.current_round = rnd
            client.model.tpLayer.adjust_loss_threshold(rnd, rounds)
            # Reset batch statistics for new round
            client.model.tpLayer.reset_batch_stats()

        for client_idx, client in enumerate(clients):
            parameters, num_examples, metrics = client.fit(
                [val.cpu().numpy() for val in global_params.values()],
                config={"local_epochs": local_epochs}
            )

            print(f"[DEBUG] phi_k_history for client {client_idx}: {client.model.tpLayer.stats.get('phi_k_history')}")
            client_deltas.append(client.weight_deltas)

            # Update statistics
            stats["flops"] += client.last_flops
            stats["memory"] = max(stats["memory"], client.last_mem)
            
            # Communication tracking
            comm_cost = 0.0
            layer_comm = {}
            for name, (indices, values) in client.weight_deltas.items():
                if isinstance(indices, tuple) and isinstance(values, torch.Tensor):
                    # Handle tuple indices (for multi-dimensional tensors)
                    total_indices = sum(idx.numel() for idx in indices)
                    total_values = values.numel()
                    layer_comm[name] = (total_indices + total_values) * 2 * 0.8
                    comm_cost += layer_comm[name]
                elif isinstance(indices, torch.Tensor) and isinstance(values, torch.Tensor):
                    # Handle single-dimensional tensors
                    indices = indices.to(torch.int32)
                    values = values.to(torch.float32)
                    layer_comm[name] = (indices.numel() + values.numel()) * 2 * 0.8
                    comm_cost += layer_comm[name]
            stats["communication"] += comm_cost
            stats["layer_communication"] = layer_comm
            
            # Update phi_k and skipping statistics
            stats["phi_k"] += get_phi_k(client.model)
            stats["skipped"] += client.model.tpLayer.stats["skipped_batches"]
            total_batches = len(client.train_loader)
            stats["effective_compute_ratio"] += 1 - (client.model.tpLayer.stats["skipped_batches"] / total_batches)
            
            # Track per-client skipped ratio with safety check
            total_batches = max(len(client.train_loader), 1)  # Ensure non-zero denominator
            skipped_ratio = client.model.tpLayer.stats.get("skipped_batches", 0) / total_batches
            stats["skipped_ratio_sum"] += skipped_ratio
            
            # Track batch-level metrics with safety checks
            if client.model.tpLayer.stats.get("phi_k_history"):
                phi_history = client.model.tpLayer.stats["phi_k_history"]
                stats["avg_phi_k"] += sum(phi_history) / max(len(phi_history), 1)
                
                loss_change_history = client.model.tpLayer.stats.get("loss_change_history", [])
                if loss_change_history:
                    stats["avg_loss_change"] += sum(loss_change_history) / max(len(loss_change_history), 1)
                
                stats["loss_threshold"] += client.model.tpLayer.stats.get("loss_threshold", 0.0)
            
            stats["sparsity"] += client.last_sparsity
            stats["grad_norm"] += client.last_avg_grad_norm
            
            for name, (indices, values) in client.weight_deltas.items():
                if isinstance(indices, torch.Tensor):
                    stats["nonzero"] += indices.numel()
            stats["total"] += sum(p.numel() for p in client.model.parameters())

        # Average the statistics across clients
        num_clients = len(clients)
        if num_clients > 0:  # Safety check
            for key in ["phi_k", "grad_norm", "sparsity", "effective_compute_ratio", 
                       "avg_phi_k", "avg_loss_change", "loss_threshold", "skipped_ratio_sum"]:
                stats[key] /= num_clients

        global_model = aggregator_fn(client_deltas, global_model, model_name, tinyprop_params, **aggregator_kwargs)

        acc = sum(
            client.local_evaluate(client.test_loader) for client in clients
        ) / len(clients)
        metrics_log["accuracy"].append(acc)
        metrics_log["flops"].append(stats.get("flops", 0.0))
        metrics_log["memory"].append(stats.get("memory", 0.0))
        metrics_log["communication"].append(stats.get("communication", 0.0))
        metrics_log["sparsity"].append(stats.get("sparsity", 0.0))
        metrics_log["avg_grad_norm"].append(stats.get("grad_norm", 0.0))
        metrics_log["phi_k"].append(stats.get("phi_k", 0.0))
        metrics_log["skipped_batches"].append(stats.get("skipped", 0))
        metrics_log["effective_compute_ratio"].append(stats.get("effective_compute_ratio", 0.0))
        metrics_log["avg_phi_k"].append(stats.get("avg_phi_k", 0.0))
        metrics_log["avg_loss_change"].append(stats.get("avg_loss_change", 0.0))
        metrics_log["loss_threshold"].append(stats.get("loss_threshold", 0.0))
        metrics_log["skipped_ratio"].append(stats.get("skipped_ratio_sum", 0.0))
        metrics_log["compression_ratio"].append(
            stats["nonzero"] / stats["total"] if stats["total"] > 0 else 0.0
        )
        metrics_log["client_eval_history"].append({cid: client.local_evaluate(client.test_loader) for cid, client in enumerate(clients)})

        if csv_log_path:
            append_to_training_log_csv(
                csv_log_path, rnd + 1, acc, stats["flops"], stats["memory"], stats["communication"],
                metrics_log["sparsity"][-1], metrics_log["avg_grad_norm"][-1], metrics_log["phi_k"][-1],
                stats["skipped"], metrics_log["effective_compute_ratio"][-1], metrics_log["compression_ratio"][-1]
            )

        if early_stopper.step(acc, rnd):
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

        # Save metrics
        if save_dir and save_interval > 0 and rnd % save_interval == 0:
            save_training_logs_csv(
                os.path.join(save_dir, f"{partition_type}_{model_name}_training_logs.csv"),
                metrics_log["accuracy"],
                metrics_log["flops"],
                metrics_log["memory"],
                metrics_log["communication"],
                metrics_log["sparsity"],
                quantization_errors
            )

        # Append detailed metrics
        append_to_training_log_csv(
            os.path.join(save_dir, f"{partition_type}_{model_name}_detailed_logs.csv"),
            rnd + 1,
            acc,
            metrics_log["flops"][-1],
            metrics_log["memory"][-1],
            metrics_log["communication"][-1],
            metrics_log["sparsity"][-1],
            metrics_log["avg_grad_norm"][-1],
            metrics_log["phi_k"][-1],
            metrics_log["skipped_batches"][-1],
            metrics_log["effective_compute_ratio"][-1],
            metrics_log["compression_ratio"][-1],
            avg_quantization_error,
            avg_scale_factor
        )

    return (
        global_model,
        metrics_log["accuracy"],
        metrics_log["flops"],
        metrics_log["memory"],
        metrics_log["communication"],
        metrics_log["sparsity"],
        metrics_log["avg_grad_norm"],
        metrics_log["phi_k"],
        metrics_log["skipped_batches"],
        metrics_log["effective_compute_ratio"],
        metrics_log["client_eval_history"],
        metrics_log["compression_ratio"],
        metrics_log["quantization_errors"],
        metrics_log["avg_scale_factors"],
        history
    )
