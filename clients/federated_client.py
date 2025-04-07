import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import flwr as fl
from models.config import get_tinyprop_config
from utils.adaptive_sparsification import AdaptiveSparsifier
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader=None, device="cpu", dataset_name="mnist"):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.cfg = get_tinyprop_config(dataset_name)
        optimizer_cfg = self.cfg["optimizer"]
        if optimizer_cfg["type"] == "sgd":
            self.optimizer = SGD(
                self.model.parameters(), 
                lr=optimizer_cfg["lr"], 
                momentum=optimizer_cfg.get("momentum", 0.9),
                weight_decay=5e-4
            )
        else:
            self.optimizer = Adam(self.model.parameters(), lr=optimizer_cfg["lr"])

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200, eta_min=optimizer_cfg["lr"] * 0.1)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize adaptive sparsifier
        self.sparsifier = AdaptiveSparsifier(
            initial_sparsity=self.cfg.get("initial_sparsity", 0.3),
            target_sparsity=self.cfg.get("target_sparsity", 0.9),
            total_rounds=self.cfg.get("total_rounds", 100),
            energy_budget=self.cfg.get("energy_budget", None)
        )

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

        self.initial_grad_norm = None
        self.smoothed_phi = None
        self.weight_deltas = {}

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):   
        state_dict = self.model.state_dict() 
        new_state_dict = {}
        for key, param in zip(state_dict.keys(), parameters):
            
            if isinstance(param, np.ndarray):
                param = torch.from_numpy(param)
            new_state_dict[key] = param.to(self.device)
        self.model.load_state_dict(new_state_dict)

    def train(self, num_epochs=1):
        
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
        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                
                self.total_batches += 1
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Compute gradient norm
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                if self.initial_grad_norm is None:
                    self.initial_grad_norm = grad_norm

                # Get adaptive sparsity from sparsifier
                current_sparsity = self.sparsifier.compute_round_sparsity(
                    round_num=self.total_batches,
                    current_accuracy=self.local_evaluate(self.train_loader) if hasattr(self, "test_loader") else 0.0
                )

                # Apply layer-wise sparsity
                layer_sparsity = self.sparsifier.compute_layer_sparsity(self.model, current_sparsity)
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in layer_sparsity:
                        sparsity = layer_sparsity[name]
                        mask = torch.rand_like(param) > sparsity
                        param.grad.data *= mask


                self.optimizer.step()

            self.scheduler.step()

        # Compute weight deltas and metrics
        self.weight_deltas = {}
        for name, param in self.model.named_parameters():
            if name in initial_state:
                delta = param.detach().cpu() - initial_state[name]
                nonzero_indices = torch.nonzero(delta.abs() > 1e-6, as_tuple=True)
                if len(nonzero_indices) > 0:
                    self.weight_deltas[name] = (nonzero_indices, delta[nonzero_indices])

        # Update metrics
        self.last_sparsity = current_sparsity
        self.last_mem = torch.cuda.max_memory_allocated(self.device)
        self.last_flops = self.cfg["full_flops_per_batch"] * (1 - current_sparsity)

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

    def fit(self, parameters):
        self.set_parameters(parameters)
        self.train(num_epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters):
        self.set_parameters(parameters)
        if hasattr(self, "test_loader"):
            acc = self.local_evaluate(self.test_loader)
            return float(acc), len(self.test_loader.dataset), {}
        else:
            raise ValueError("No test dataset provided for evaluation.")