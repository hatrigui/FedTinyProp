import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import flwr as fl
from models.config import get_config

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data=None, device="cpu", dataset_name="mnist"):
        self.device = device
        self.model = model.to(device)
        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.test_data = test_data
        if test_data is not None:
            self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Load dataset-specific config
        cfg = get_config(dataset_name)

        if cfg["optimizer"] == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
        else:
            self.optimizer = Adam(self.model.parameters(), lr=cfg["lr"])

        self.criterion = nn.CrossEntropyLoss()

        self.last_flops = 0.0
        self.last_mem = 0.0
        self.last_comm = 0.0
        self.last_sparsity = 0.0

        self.initial_grad_norm = 1e-9
        self.did_init_grad = False
        self.skip_threshold = cfg["skip_threshold"]
        self.full_flops_per_batch = cfg["full_flops_per_batch"]

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {}
        for key, param in zip(state_dict.keys(), parameters):
            new_state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(new_state_dict)

    def train(self, num_epochs=1):
        self.last_flops = 0.0
        self.last_mem = 0.0
        self.last_comm = 0.0
        self.last_sparsity = 0.0

        total_nonzero_grads = 0
        total_grads = 0
        total_changed_params = 0
        peak_nonzero = 0

        self.model.train()
        for _ in range(num_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                total_gnorm = 0.0
                total_params = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        g = param.grad.data
                        total_gnorm += g.norm(2).item()
                        total_params += 1

                if total_params > 0:
                    avg_gnorm = total_gnorm / total_params
                else:
                    avg_gnorm = 0.0

                if not self.did_init_grad:
                    self.initial_grad_norm = max(1e-9, avg_gnorm)
                    self.did_init_grad = True

                if total_gnorm < self.skip_threshold:
                    continue

                phi = avg_gnorm / self.initial_grad_norm

                self.model.adaptive_ratio = phi

                nonzero = 0
                total_elems = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        g = param.grad.data
                        nz = (g.abs() > 1e-9).sum().item()
                        nonzero += nz
                        total_elems += g.numel()

                total_nonzero_grads += nonzero
                total_grads += total_elems

                changed_params = nonzero
                total_changed_params += changed_params

                self.optimizer.step()

                fraction_sparsity = nonzero / (total_elems if total_elems > 0 else 1)
                self.last_flops += self.full_flops_per_batch * fraction_sparsity

                if nonzero > peak_nonzero:
                    peak_nonzero = nonzero

        if total_grads > 0:
            self.last_sparsity = total_nonzero_grads / total_grads
        self.last_mem = peak_nonzero * 4.0
        self.last_comm = total_changed_params * 4.0

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
            acc = self.local_evaluate(self.test_loader)
            return float(acc), len(self.test_loader.dataset), {}
        else:
            raise ValueError("No test dataset provided for evaluation.")