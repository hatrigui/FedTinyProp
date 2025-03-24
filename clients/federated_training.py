import torch
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient
from models.model import get_tinyprop_model

def aggregate_models(model_list, model_name):
    """
    Aggregate models by averaging their parameters.
    """
    assert model_name in ['mnist', 'fashionmnist', 'cifar10', 'cifar100'], f"[DEBUG] Invalid model_name passed to aggregate_models(): {model_name}"
    
    global_model = get_tinyprop_model(model_name)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [model.state_dict()[key].float() for model in model_list], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def federated_training(client_datasets, model_name, testset, rounds=5, device="cpu"):
    """
    Run federated training on the given client datasets using the specified model.

    Args:
        client_datasets: List of dataset subsets (one per client).
        model_name (str): Model name (e.g., 'mnist').
        testset: The test dataset.
        rounds (int): Number of federated rounds.
        device (str): 'cpu' or 'cuda'.

    Returns:
        global_model: The aggregated global model.
        test_accs: List of test accuracies per round.
    """
    clients = [
        FederatedClient(get_tinyprop_model(model_name), dataset, device=device)
        for dataset in client_datasets
    ]
    
    global_model = get_tinyprop_model(model_name).to(device)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    test_accs = []

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_models = []

        for client in clients:
            client.set_parameters([val.cpu().numpy() for val in global_params.values()])
            client.train(num_epochs=1)
            client_models.append(client.model)

        global_model = aggregate_models(client_models, model_name)

        # Use the global model directly to evaluate on testset
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
