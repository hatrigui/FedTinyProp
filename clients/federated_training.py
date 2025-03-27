import torch
from models.model import get_tinyprop_model
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient

def federated_training(
    client_datasets,
    model_name,
    testset,
    tinyprop_params,
    aggregator_fn,
    aggregator_kwargs=None,
    rounds=5,
    device="cpu",
    local_epochs=1   
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
    test_accs = []

    for rnd in range(rounds):
        print(f"Round {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_models = []

        for client in clients:
            client.set_parameters([val.cpu().numpy() for val in global_params.values()])
            client.train(num_epochs=local_epochs)  
            client_models.append(client.model)

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
        test_accs.append(acc)
        print(f"Test Accuracy: {acc:.4f}")

    return global_model, test_accs
