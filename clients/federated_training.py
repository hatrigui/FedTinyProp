from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from models.model import get_tinyprop_model 
from models.config import get_tinyprop_config
from utils.early_stopping import EarlyStoppingMonitor
from utils.save_results import append_to_training_log_csv
from utils.tinyml_metrics import TinyMLMetrics
from utils.adaptive_sparsification import AdaptiveSparsifier


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
    energy_budget: Optional[float] = None
):

    if aggregator_kwargs is None:
        aggregator_kwargs = {}

    if "dataset_sizes" not in aggregator_kwargs:
        aggregator_kwargs["dataset_sizes"] = [len(ds) for ds in client_datasets]

    config = get_tinyprop_config(model_name)

    # Initialize global model first
    global_model = get_tinyprop_model(model_name, tinyprop_params)
    global_model = global_model.to(device)
    
    # Initialize clients with the same model architecture
    clients = []
    for i, dataset in enumerate(client_datasets):
        # Create a new model instance for each client
        client_model = get_tinyprop_model(model_name, tinyprop_params)
        
        # Initialize with global model parameters
        client_model.load_state_dict(global_model.state_dict())
        client_model = client_model.to(device)
        client = FederatedClient(
            client_model,
            DataLoader(dataset, batch_size=32, shuffle=True),
            test_loader=DataLoader(testset, batch_size=32, shuffle=False),
            device=device,
            dataset_name=model_name
        )
        clients.append(client)

    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    accuracy_list = []
    flops_list    = []
    mem_list      = []
    comm_list     = []
    sparsity_list = []
    avg_grad_norm_list = []
    avg_phi_list = []
    skipped_batches_list = []
    effective_compute_ratio_list = []
    client_eval_history = []
    compression_ratio_list = []

    early_stopper = EarlyStoppingMonitor(patience=early_stopping_patience, delta=early_stopping_delta)

    # Initialize metrics tracking
    metrics = TinyMLMetrics()
    
    # Initialize adaptive sparsification
    sparsifier = AdaptiveSparsifier(
        initial_sparsity=initial_sparsity,
        target_sparsity=target_sparsity,
        total_rounds=rounds,
        energy_budget=energy_budget
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'sparsity': [],
        'energy': [],
        'communication': []
    }

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_sparse_deltas = []

        round_flops = 0.0
        round_mem   = 0.0
        round_comm  = 0.0
        round_sparsity = 0.0
        round_grad_norm = 0.0
        round_phi = 0.0
        round_skipped_batches = 0
        round_nonzero_weights = 0
        round_total_weights = 0
        client_local_accuracies = {}
        local_steps = []

        for cid, client in enumerate(clients):
            client.set_parameters([val.cpu().numpy() for val in global_params.values()])
            client.train(num_epochs=local_epochs)

            sparse_delta = client.weight_deltas
            client_sparse_deltas.append(sparse_delta)

            round_flops += client.last_flops
            round_mem = max(round_mem, client.last_mem)
            
            # Calculate communication cost with TinyML optimizations
            sparse_comm = 0
            for values, indices in sparse_delta.values():
                values_size = len(values) * 2  # float16
                indices_size = len(indices) * 2  # uint16
                sparse_comm += (values_size + indices_size) * 0.8
            
            round_comm += sparse_comm
            round_sparsity += client.last_sparsity
            round_grad_norm += client.last_avg_grad_norm
            round_phi += client.last_phi
            round_skipped_batches += client.num_skipped_batches
            round_nonzero_weights += sum(len(i) for _, i in sparse_delta.values())
            round_total_weights += sum(torch.numel(p) for p in client.model.parameters())

            if hasattr(client, "test_loader"):
                acc = client.local_evaluate(client.test_loader)
                client_local_accuracies[cid] = acc

            local_steps.append(local_epochs)

        aggregator_kwargs["global_params"] = global_params
        aggregator_kwargs["local_steps"] = local_steps

        global_model = aggregator_fn(
            client_sparse_deltas,
            global_model,
            model_name,
            tinyprop_params,
            **aggregator_kwargs
        )

        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        accuracy_list.append(acc)
        flops_list.append(round_flops)
        mem_list.append(round_mem)
        comm_list.append(round_comm)
        mean_sparsity = round_sparsity / len(clients) if clients else 0.0
        sparsity_list.append(mean_sparsity)

        mean_grad_norm = round_grad_norm / len(clients) if clients else 0.0
        mean_phi = round_phi / len(clients) if clients else 0.0
        avg_grad_norm_list.append(mean_grad_norm)
        avg_phi_list.append(mean_phi)

        skipped_batches_list.append(round_skipped_batches)
        compression_ratio = round_nonzero_weights / round_total_weights if round_total_weights > 0 else 0.0
        compression_ratio_list.append(compression_ratio)

        full_flops = len(clients) * len(client.train_loader) * config["full_flops_per_batch"]
        effective_compute_ratio = round_flops / full_flops if full_flops > 0 else 0.0
        effective_compute_ratio_list.append(effective_compute_ratio)

        client_eval_history.append(client_local_accuracies)

        print(f"Test Accuracy: {acc:.4f}")
        print(f"[Compute] round_flops={round_flops:.2f}, [Mem] peak={round_mem} bytes, [Comm] {round_comm} bytes, [Sparsity] {mean_sparsity*100:.2f}%")

        if csv_log_path:
            append_to_training_log_csv(
                csv_log_path,
                round_num=rnd + 1,
                accuracy=acc,
                flops=round_flops,
                memory_bytes=round_mem,
                communication_bytes=round_comm,
                sparsity=mean_sparsity,
                avg_grad_norm=mean_grad_norm,
                avg_phi=mean_phi,
                skipped_batches=round_skipped_batches,
                effective_compute_ratio=effective_compute_ratio,
                compression_ratio=compression_ratio
            )

        if early_stopper.step(acc, rnd):
            print(f"\n[Early Stop] Triggered after {rnd+1} rounds!")
            print(f"Best Accuracy: {early_stopper.best_acc:.4f} at Round {early_stopper.best_round + 1}")
            break

        # Compute current sparsity
        current_sparsity = sparsifier.compute_round_sparsity(
            rnd,
            accuracy_list[-1] if accuracy_list else 0,
            history['energy'][-1] if history['energy'] else None
        )
        
        # Track metrics
        round_train_loss = 0.0
        round_train_correct = 0
        round_train_total = 0
        
        for client in clients:
            client.model.train()
            for images, labels in client.train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = client.model(images)
                loss = client.criterion(outputs, labels)
                round_train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                round_train_correct += (predicted == labels).sum().item()
                round_train_total += labels.size(0)
        
        avg_train_loss = round_train_loss / len(clients)
        train_accuracy = round_train_correct / round_train_total if round_train_total > 0 else 0.0
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_accuracy'].append(acc)
        history['sparsity'].append(current_sparsity)
        
        # Compute and track energy consumption
        round_energy = compute_round_energy(clients, current_sparsity)
        history['energy'].append(round_energy)
        
        # Compute and track communication cost
        round_comm = compute_round_communication(clients, current_sparsity)
        history['communication'].append(round_comm)
        
        # Print round summary
        print(f"Round {rnd + 1}/{rounds}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Sparsity: {mean_sparsity:.4f}")
        print(f"Energy: {round_energy:.4f} J")
        print(f"Communication: {round_comm/1024:.2f} KB")
        print("-" * 50)

    return global_model, accuracy_list, flops_list, mem_list, comm_list, sparsity_list, avg_grad_norm_list, avg_phi_list, skipped_batches_list, effective_compute_ratio_list, client_eval_history, compression_ratio_list, history

def aggregate_model_parameters(clients: List[FederatedClient]) -> None:
    """Aggregate model parameters using FedAvg."""
    # Get the first client's model parameters
    global_params = {}
    for name, param in clients[0].model.named_parameters():
        global_params[name] = param.data.clone()
    
    # Average parameters across clients
    for name in global_params:
        for client in clients[1:]:
            global_params[name] += client.model.state_dict()[name]
        global_params[name] /= len(clients)
    
    # Update all clients with averaged parameters
    for client in clients:
        for name, param in client.model.named_parameters():
            param.data = global_params[name].clone()

def compute_round_energy(clients: List[FederatedClient], sparsity: float) -> float:
    """Compute total energy consumption for one round."""
    total_energy = 0
    for client in clients:
        # Compute FLOPs
        compute_flops = sum(p.numel() for p in client.model.parameters()) * 2  # forward + backward
        compute_flops *= (1 - sparsity)  # account for sparsity
        
        # Compute memory access
        memory_bytes = sum(p.numel() * p.element_size() for p in client.model.parameters())
        memory_bytes *= (1 - sparsity)  # account for sparsity
        
        # Compute communication bytes
        communication_bytes = compute_round_communication([client], sparsity)
        
        # Use TinyMLMetrics to estimate energy
        metrics = TinyMLMetrics()
        energy_metrics = metrics.estimate_energy_consumption(
            compute_flops,
            memory_bytes,
            communication_bytes
        )
        total_energy += energy_metrics.total_energy
    
    return total_energy

def compute_round_communication(clients: List[FederatedClient], sparsity: float) -> float:
    """Compute total communication cost for one round."""
    total_bytes = 0
    for client in clients:
        # Get model parameters
        params_size = sum(p.numel() * 2 for p in client.model.parameters())  # 2 bytes per value (fp16)
        
        # Account for sparsity and indices
        sparse_size = params_size * (1 - sparsity)  # Only send non-zero values
        index_size = sparse_size  # 2 bytes per index
        
        # Add compression benefit (assume 20% reduction from pattern-based compression)
        compressed_size = (sparse_size + index_size) * 0.8
        
        total_bytes += compressed_size
    
    return total_bytes