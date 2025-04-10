from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from models.model import get_tinyprop_model 
from models.config import get_tinyprop_config
from utils.early_stopping import EarlyStoppingMonitor
from utils.save_results import append_to_training_log_csv

from utils.adaptive_sparsification import AdaptiveSparsifier
from models.model import get_tinyprop_model
from clients.aggregators import sparse_fedavg_aggregate

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
    alpha=None
):
    if aggregator_kwargs is None:
        aggregator_kwargs = {}

    if "dataset_sizes" not in aggregator_kwargs:
        aggregator_kwargs["dataset_sizes"] = [len(ds) for ds in client_datasets]

    config = get_tinyprop_config(model_name)
    global_model = get_tinyprop_model(model_name, tinyprop_params)

    if num_clients is not None and num_rounds is not None and partition_type is not None and alpha is not None:
        print(f"\n[Training Debug] Starting federated training with {num_clients} clients")
        print(f"[Training Debug] Partition type: {partition_type}, alpha: {alpha}")
        
        # Initialize global model and clients
        global_model = get_tinyprop_model(model_name, tinyprop_params)
        global_params = global_model.state_dict()
        clients = []
        dataset_sizes = []
        
        print("\n[Training Debug] Initializing clients...")
        for i in range(num_clients):
            # Create proper DataLoaders
            train_loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)
            test_loader = DataLoader(testset, batch_size=32, shuffle=False)
            dataset_sizes.append(len(train_loader.dataset))
            
            # Initialize client with proper loaders
            client = FederatedClient(
                model=get_tinyprop_model(model_name, tinyprop_params),
                train_loader=train_loader,
                test_loader=test_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dataset_name="fashionmnist"
            )
            clients.append(client)
            print(f"[Training Debug] Client {i} initialized with {dataset_sizes[i]} samples")
        
        # Training loop
        client_deltas = []
        for round_num in range(num_rounds):
            print(f"\n[Training Debug] Starting round {round_num + 1}/{num_rounds}")
            
            # Train each client
            for client_idx, client in enumerate(clients):
                print(f"\n[Training Debug] Training client {client_idx}")
                try:
                    parameters, num_examples, metrics = client.fit(
                        [val.cpu().numpy() for val in global_params.values()],
                        config={"local_epochs": local_epochs}
                    )
                    
                    # Debug weight deltas
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
                    print(f"[Training Error] Error in client {client_idx}: {str(e)}")
                    raise  # Re-raise the exception to see the full traceback
            
            # Aggregate updates
            print("\n[Training Debug] Aggregating client updates...")
            global_model = sparse_fedavg_aggregate(
                sparse_updates=client_deltas,
                global_model=global_model,
                model_name=model_name,
                tinyprop_params=tinyprop_params,
                dataset_sizes=dataset_sizes
            )
            global_params = global_model.state_dict()
            
            # Evaluate global model
            print("\n[Training Debug] Evaluating global model...")
            test_loader = DataLoader(testset, batch_size=64, shuffle=False)
            global_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(global_model.device), labels.to(global_model.device)
                    outputs = global_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"[Training Debug] Round {round_num + 1} accuracy: {accuracy:.2f}%")
            
            # Clear client deltas for next round
            client_deltas = []
        
        return global_model

    global_model = get_tinyprop_model(model_name, tinyprop_params).to(device)

    clients = []
    for dataset in client_datasets:
        client_model = get_tinyprop_model(model_name, tinyprop_params).to(device)
        client_model.load_state_dict(global_model.state_dict())
        clients.append(FederatedClient(
            client_model,
            DataLoader(dataset, batch_size=32, shuffle=True),
            test_loader=DataLoader(testset, batch_size=32, shuffle=False),
            device=device,
            dataset_name=model_name
        ))

    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    metrics_log = {
        "accuracy": [], "flops": [], "memory": [], "communication": [], "sparsity": [],
        "avg_grad_norm": [], "avg_phi": [], "skipped_batches": [],
        "effective_compute_ratio": [], "compression_ratio": [], "client_eval_history": []
    }
    early_stopper = EarlyStoppingMonitor(patience=early_stopping_patience, delta=early_stopping_delta)
    sparsifier = AdaptiveSparsifier(initial_sparsity, target_sparsity, rounds, energy_budget=energy_budget)
    history = {k: [] for k in ["train_loss", "train_accuracy", "test_accuracy", "sparsity", "energy", "communication"]}

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_deltas = []
        stats = {
            "flops": 0.0, "memory": 0.0, "communication": 0.0, "sparsity": 0.0,
            "grad_norm": 0.0, "phi": 0.0, "skipped": 0, "nonzero": 0, "total": 0
        }

        for client in clients:
            parameters, num_examples, metrics = client.fit(
                [val.cpu().numpy() for val in global_params.values()],
                config={"local_epochs": local_epochs}
            )
            client_deltas.append(client.weight_deltas)

            stats["flops"] += client.last_flops
            stats["memory"] = max(stats["memory"], client.last_mem)
            
            comm_cost = 0.0
            for name, (indices, values) in client.weight_deltas.items():
                if isinstance(indices, torch.Tensor) and isinstance(values, torch.Tensor):
                    comm_cost += (indices.numel() + values.numel()) * 2 * 0.8
            stats["communication"] += comm_cost
            
            stats["sparsity"] += client.last_sparsity
            stats["grad_norm"] += client.last_avg_grad_norm
            stats["phi"] += client.last_phi
            stats["skipped"] += client.num_skipped_batches
            
            for name, (indices, values) in client.weight_deltas.items():
                if isinstance(indices, torch.Tensor):
                    stats["nonzero"] += indices.numel()
            stats["total"] += sum(p.numel() for p in client.model.parameters())

        global_model = aggregator_fn(client_deltas, global_model, model_name, tinyprop_params, **aggregator_kwargs)

        acc = sum(
            client.local_evaluate(client.test_loader) for client in clients
        ) / len(clients)
        metrics_log["accuracy"].append(acc)
        metrics_log["flops"].append(stats["flops"])
        metrics_log["memory"].append(stats["memory"])
        metrics_log["communication"].append(stats["communication"])
        metrics_log["sparsity"].append(stats["sparsity"] / len(clients))
        metrics_log["avg_grad_norm"].append(stats["grad_norm"] / len(clients))
        metrics_log["avg_phi"].append(stats["phi"] / len(clients))
        metrics_log["skipped_batches"].append(stats["skipped"])
        metrics_log["effective_compute_ratio"].append(
            stats["flops"] / (len(clients) * len(client.train_loader) * config["full_flops_per_batch"])
        )
        metrics_log["compression_ratio"].append(
            stats["nonzero"] / stats["total"] if stats["total"] > 0 else 0.0
        )
        metrics_log["client_eval_history"].append({cid: client.local_evaluate(client.test_loader) for cid, client in enumerate(clients)})

        if csv_log_path:
            append_to_training_log_csv(
                csv_log_path, rnd + 1, acc, stats["flops"], stats["memory"], stats["communication"],
                metrics_log["sparsity"][-1], metrics_log["avg_grad_norm"][-1], metrics_log["avg_phi"][-1],
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

    return (
        global_model,
        metrics_log["accuracy"],
        metrics_log["flops"],
        metrics_log["memory"],
        metrics_log["communication"],
        metrics_log["sparsity"],
        metrics_log["avg_grad_norm"],
        metrics_log["avg_phi"],
        metrics_log["skipped_batches"],
        metrics_log["effective_compute_ratio"],
        metrics_log["client_eval_history"],
        metrics_log["compression_ratio"],
        history
    )
