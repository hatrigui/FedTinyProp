import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from models.model import get_tinyprop_model
from utils.performance_visualizations import plot_fed_metrics  

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
    visualize=True
):
   
    if aggregator_kwargs is None:
        aggregator_kwargs = {}

    if "dataset_sizes" not in aggregator_kwargs:
        aggregator_kwargs["dataset_sizes"] = [len(ds) for ds in client_datasets]

    clients = [
        FederatedClient(
            get_tinyprop_model(model_name, tinyprop_params),
            dataset,
            device=device
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

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_models = []

        round_flops = 0.0
        round_mem   = 0.0
        round_comm  = 0.0
        round_sparsity = 0.0

        for client in clients:
            client.set_parameters([val.cpu().numpy() for val in global_params.values()])
            client.train(num_epochs=local_epochs)
            client_models.append(client.model)

            round_flops    += client.last_flops
            if client.last_mem > round_mem:
                round_mem = client.last_mem 
            round_comm     += client.last_comm
            round_sparsity += client.last_sparsity

        global_model = aggregator_fn(
            client_models,
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
        mean_sparsity = round_sparsity / len(clients) if len(clients) > 0 else 0
        sparsity_list.append(mean_sparsity)

        print(f"Test Accuracy: {acc:.4f}")
        print(f"[Compute] round_flops={round_flops:.2f}, [Mem] peak={round_mem} bytes, [Comm] {round_comm} bytes, [Sparsity] {mean_sparsity*100:.2f}%")

    if visualize:
        plot_fed_metrics(accuracy_list, flops_list, mem_list, comm_list, sparsity_list)

    return global_model, accuracy_list, flops_list, mem_list, comm_list, sparsity_list
