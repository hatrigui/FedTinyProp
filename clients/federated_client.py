import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import flwr as fl
from models.config import get_tinyprop_config
from utils.training_helpers import (
    compute_avg_grad_norm,
    compute_adaptive_ratio,
    compute_sparsity_and_flops,
    compute_sparse_deltas
)

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data=None, device="cpu", dataset_name="mnist"):
        self.device = device
        self.model = model.to(device)
        self.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.test_data = test_data
        if test_data is not None:
            self.test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        self.cfg = get_tinyprop_config(dataset_name)
        optimizer_cfg = self.cfg["optimizer"]
        if optimizer_cfg["type"] == "sgd":
            self.optimizer = SGD(
                self.model.parameters(), 
                lr=optimizer_cfg["lr"], 
                momentum=optimizer_cfg.get("momentum", 0.0)
            )
        else:
            self.optimizer = Adam(self.model.parameters(), lr=optimizer_cfg["lr"])

        self.criterion = nn.CrossEntropyLoss()

        self.skip_threshold = self.cfg["skip_threshold"]
        self.full_flops_per_batch = self.cfg["full_flops_per_batch"]
        self.phi_min = self.cfg.get("phi_min", 0.0)

        self.last_flops = 0.0
        self.last_mem = 0.0
        self.last_comm = 0.0
        self.last_sparsity = 0.0
        self.last_avg_grad_norm = 0.0
        self.last_phi = 0.0
        self.num_skipped_batches = 0
        self.total_batches = 0
        self.compression_ratio = 0.0
        self.layer_sparsity = {}

        self.initial_grad_norm = 1e-9
        self.did_init_grad = False
        self.weight_deltas = {}

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {}
        for key, param in zip(state_dict.keys(), parameters):
            new_state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(new_state_dict)

    def train(self, num_epochs=1, warmup_rounds=5, current_round=0):
        self.last_flops = 0.0
        self.last_mem = 0.0
        self.last_comm = 0.0
        self.last_sparsity = 0.0
        self.last_avg_grad_norm = 0.0
        self.last_phi = 0.0
        self.num_skipped_batches = 0
        self.total_batches = 0
        self.weight_deltas = {}

        initial_state = {
            name: param.detach().clone().cpu()
            for name, param in self.model.state_dict().items()
        }

        self.model.train()
        for images, labels in self.train_loader:
            self.total_batches += 1
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            avg_gnorm = compute_avg_grad_norm(self.model)
            self.last_avg_grad_norm = avg_gnorm

            if not self.did_init_grad:
                self.initial_grad_norm = max(1e-9, avg_gnorm)
                self.did_init_grad = True

            if current_round >= warmup_rounds and avg_gnorm < self.skip_threshold:
                self.num_skipped_batches += 1
                continue

            if current_round >= warmup_rounds:
                phi = compute_adaptive_ratio(avg_gnorm, self.initial_grad_norm, self.phi_min)
                self.last_phi = phi
                self.model.adaptive_ratio = phi

            self.optimizer.step()

            batch_sparsity, batch_flops = compute_sparsity_and_flops(self.model, self.full_flops_per_batch)
            self.last_flops += batch_flops

        self.weight_deltas, sparsity, peak_mem = compute_sparse_deltas(self.model, initial_state, self.device)
        self.last_sparsity = sparsity
        self.last_mem = peak_mem

        # Compression ratio
        total_weights = sum(p.numel() for p in self.model.parameters())
        nonzero_weights = sum(len(i) for _, i in self.weight_deltas.values())
        self.compression_ratio = nonzero_weights / total_weights if total_weights > 0 else 0.0

        # Layer-wise sparsity (optional detail)
        for name, param in self.model.named_parameters():
            if name in initial_state:
                delta = param.detach().cpu() - initial_state[name]
                total = delta.numel()
                nonzero = (delta.abs() > 1e-9).sum().item()
                self.layer_sparsity[name] = 1.0 - (nonzero / total) if total > 0 else 1.0

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