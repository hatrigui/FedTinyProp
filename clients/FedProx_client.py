import torch
import torch.nn.functional as F
from clients.federated_client import FederatedClient

class FederatedClientWithFedProx(FederatedClient):
    def __init__(self, model, train_data, test_data=None, device="cpu", mu=0.1, global_model=None):
        super().__init__(model, train_data, test_data, device)
        self.mu = mu
        self.global_model = global_model

    def set_global_model(self, global_model):
        self.global_model = global_model

    def train(self, num_epochs=1):
        self.model.train()
        global_params = list(self.global_model.parameters())
        for _ in range(num_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Standard loss
                loss = self.criterion(outputs, labels)

                # FedProx proximal regularization
                prox_loss = 0.0
                for param, global_param in zip(self.model.parameters(), global_params):
                    prox_loss += ((param - global_param)**2).sum()
                prox_loss = (self.mu / 2) * prox_loss

                # Total Loss
                total_loss = loss + prox_loss
                total_loss.backward()
                self.optimizer.step()
