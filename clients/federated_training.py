import torch
from models.model import get_tinyprop_model
from torch.utils.data import DataLoader
from clients.federated_client import FederatedClient


def aggregate_models(model_list, model_name, tinyprop_params):
    """
    Aggregate models by averaging their parameters. 

    """
    global_model = get_tinyprop_model(model_name, tinyprop_params)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [model.state_dict()[key].float() for model in model_list], 0
        ).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model

def federated_training(client_datasets, model_name, testset, tinyprop_params, rounds=5, device="cpu"):
    """
    Run federated training with a given set of tinyprop_params.
    """

    clients = [
        FederatedClient(get_tinyprop_model(model_name, tinyprop_params), dataset, device=device)
        for dataset in client_datasets
    ]

    global_model = get_tinyprop_model(model_name, tinyprop_params).to(device)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    test_accs = []

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_models = []

        for client in clients:
            # Send global params to client
            client.set_parameters([val.cpu().numpy() for val in global_params.values()])
            # Local training
            client.train(num_epochs=1)
            client_models.append(client.model)

        # Aggregate updates
        global_model = aggregate_models(client_models, model_name, tinyprop_params)

        # Evaluate the new global model
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
