import torch
import torch.nn.functional as F
from clients.federated_client import FederatedClient

class FederatedClientWithFedProx(FederatedClient):
    def __init__(self, model, train_data, test_data=None, device="cpu", mu=0.1, global_model=None):
        super().__init__(model, train_data, test_data, device)
        self.mu = mu  # Proximal term coefficient
        self.global_model = global_model  # Ensure it's assigned here as an attribute

    def train(self, num_epochs=1):
        self.model.train()
        for _ in range(num_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Standard cross-entropy loss
                loss = self.criterion(outputs, labels)

                # FedProx term: (mu / 2) * ||local model parameters - global model parameters||^2
                prox_loss = 0
                global_params = self.global_model.state_dict()  # Now this should work
                for param, global_param in zip(self.model.parameters(), global_params.values()):
                    prox_loss += torch.sum((param - global_param) ** 2)
                prox_loss = (self.mu / 2) * prox_loss

                # Final loss = local loss + FedProx regularization
                total_loss = loss + prox_loss
                total_loss.backward()
                self.optimizer.step()
