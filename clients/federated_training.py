import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from models.model import get_tinyprop_model 
from models.config import get_tinyprop_config
from utils.early_stopping import EarlyStoppingMonitor
from utils.save_results import append_to_training_log_csv

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
    warmup_rounds=5,
    json_log_path=None
):
    if aggregator_kwargs is None:
        aggregator_kwargs = {}

    if "dataset_sizes" not in aggregator_kwargs:
        aggregator_kwargs["dataset_sizes"] = [len(ds) for ds in client_datasets]

    config = get_tinyprop_config(model_name)

    clients = [
        FederatedClient(
            get_tinyprop_model(model_name, tinyprop_params),
            dataset,
            device=device,
            dataset_name=model_name
        )
        for dataset in client_datasets
    ]

    global_model = get_tinyprop_model(model_name, tinyprop_params).to(device)
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
            client.train(num_epochs=local_epochs, warmup_rounds=warmup_rounds, current_round=rnd)

            sparse_delta = client.weight_deltas
            client_sparse_deltas.append(sparse_delta)

            round_flops += client.last_flops
            round_mem = max(round_mem, client.last_mem)
            round_comm += sum(len(v) * 4 + len(i) * 4 for v, i in sparse_delta.values())
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

        if json_log_path:
            append_to_training_log_csv(
                json_log_path,
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

    return global_model, accuracy_list, flops_list, mem_list, comm_list, sparsity_list, avg_grad_norm_list, avg_phi_list, skipped_batches_list, effective_compute_ratio_list, client_eval_history, compression_ratio_list