def weighted_aggregate_models(model_list, dataset_sizes, model_name):
    """
    Aggregate models by averaging their parameters, weighted by dataset size.
    
    Args:
        model_list: List of local models from each client.
        dataset_sizes: List of integers (length of each client's dataset).
        model_name: e.g., 'mnist'
    """
    from models.model import get_tinyprop_model
    import torch

    # Create the same model architecture to combine weights into
    global_model = get_tinyprop_model(model_name)
    global_dict = global_model.state_dict()

    total_data = sum(dataset_sizes)  # total number of samples across all clients

    # Weighted average each layer by client data size
    for key in global_dict.keys():
        accum = torch.zeros_like(global_dict[key], dtype=torch.float)
        for model, size in zip(model_list, dataset_sizes):
            accum += model.state_dict()[key] * (size / total_data)
        global_dict[key] = accum

    global_model.load_state_dict(global_dict)
    return global_model


def weighted_federated_training(client_datasets, model_name, testset, rounds=5, device="cpu"):
    from clients.federated_client import FederatedClient
    from models.model import get_tinyprop_model
    from torch.utils.data import DataLoader
    import torch

    # 1) Compute each client's dataset size
    dataset_sizes = [len(ds) for ds in client_datasets]

    # 2) Create clients
    clients = [
        FederatedClient(get_tinyprop_model(model_name), dataset, device=device)
        for dataset in client_datasets
    ]

    # 3) Initialize global model
    global_model = get_tinyprop_model(model_name).to(device)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    test_accs = []

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_models = []

        # 4) Each client trains locally
        for client in clients:
            client.set_parameters([val.cpu().numpy() for val in global_params.values()])
            client.train(num_epochs=1)
            client_models.append(client.model)

        # 5) Weighted aggregation
        global_model = weighted_aggregate_models(client_models, dataset_sizes, model_name)

        # 6) Evaluate the new global model
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

