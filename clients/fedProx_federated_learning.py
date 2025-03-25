from clients.FedProx_client import FederatedClientWithFedProx
from models.model import get_tinyprop_model
from clients.federated_training import aggregate_models
from torch.utils.data import DataLoader
import torch

def federated_training_with_fedprox(client_datasets, model_name, testset, rounds=5, device="cpu", mu=0.1):

    global_model = get_tinyprop_model(model_name).to(device)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    test_accs = []

    clients = [
        FederatedClientWithFedProx(get_tinyprop_model(model_name), dataset, device=device, mu=mu)
        for dataset in client_datasets
    ]

    for rnd in range(rounds):
        print(f"\nRound {rnd+1}/{rounds}")
        global_params = global_model.state_dict()
        client_models = []

        for client in clients:
            client.set_parameters([val.cpu().numpy() for val in global_params.values()])
            client.set_global_model(global_model)  # Important fix here
            client.train(num_epochs=1)
            client_models.append(client.model)

        global_model = aggregate_models(client_models, model_name)

        # Evaluate global model
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
