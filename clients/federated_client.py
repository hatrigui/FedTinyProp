import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import flwr as fl

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data=None, device="cpu"):
        """
        Initialize the FederatedClient.
        
        Args:
            model: A PyTorch model.
            train_data: Training dataset (or subset) for this client.
            test_data: Optional test dataset to be used during evaluation.
            device (str): Device to use (e.g., "cpu" or "cuda").
        """
        self.device = device
        self.model = model.to(device)
        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.test_data = test_data  
        if test_data is not None:
            self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {}
        for key, param in zip(state_dict.keys(), parameters):
            new_state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(new_state_dict)

    def train(self, num_epochs=1):
        self.model.train()
        for _ in range(num_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def local_evaluate(self, data_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(num_epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if hasattr(self, "test_loader"):
            test_loader = self.test_loader
        else:
            raise ValueError("No test dataset provided for evaluation.")
        accuracy = self.local_evaluate(test_loader)
        return float(accuracy), len(test_loader.dataset), {}
