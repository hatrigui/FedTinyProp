import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import flwr as fl
import numpy as np
from typing import Dict, Tuple
import os, psutil
import pandas as pd

from utils.flops_calculator import compute_model_flops
from utils.memory_calculator import estimate_model_memory

class FederatedDenseClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, model: nn.Module, train_loader, test_loader, cfg: dict, device: str = None):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device
        self.optimizer = SGD(
            self.model.parameters(),
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"].get("momentum", 0.9),
            weight_decay=cfg["optimizer"].get("weight_decay", 0.0),
            nesterov=cfg["optimizer"].get("nesterov", False),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["lr_scheduler"]["T_max"],
            eta_min=cfg["lr_scheduler"]["eta_min"]
        ) if "lr_scheduler" in cfg else None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
        self.quantization_enabled = cfg.get("quantization", {}).get("enabled", False)
        self.quantization_bits = cfg.get("quantization", {}).get("bits", 32)
        self.last_flops = 0.0
        self.last_mem = 0.0
        self.last_comm = 0.0
        self.last_sparsity = 0.0
        self.communication_metrics = {}

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), parameters):
            param = torch.from_numpy(param) if isinstance(param, np.ndarray) else param
            state_dict[key] = param.to(self.device)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        batch_size = config.get("batch_size", 32)
        local_epochs = config.get("local_epochs", 1)
        loss, acc = self.train(local_epochs, batch_size)
        return self.get_parameters(), len(self.train_loader.dataset), {
            "loss": float(loss),
            "accuracy": float(acc),
            "phi": 1.0,
            "skipped_batches": 0,
            "sparsity": 0.0,
            "flops": float(self.last_flops),
            "memory_MB": float(self.last_mem),
            "communication_bytes": float(self.last_comm),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return float(self.local_evaluate()), len(self.test_loader.dataset)

    def train(self, epochs: int, batch_size: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total, total_flops = 0.0, 0, 0, 0.0
        sample_input, _ = next(iter(self.train_loader))
        layer_sparsity = {name: 0.0 for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
        
        # Store initial model size for communication tracking
        model_size_bytes = sum(p.numel() * 4 for p in self.model.parameters())
        
        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
                batch_flops, _ = compute_model_flops(self.model, x.shape, layer_sparsity)
                total_flops += batch_flops
            if self.scheduler: self.scheduler.step()
        
        self.last_flops = total_flops
        mem_report = estimate_model_memory(self.model, batch_size=sample_input.shape[0], input_shape=sample_input.shape[1:])
        self.last_mem = mem_report["total_MB"]
        
        # Calculate communication costs:
        # 1. Download: Full model weights (4 bytes per parameter)
        # 2. Upload: Full model weights (4 bytes per parameter)
        download_bytes = model_size_bytes
        upload_bytes = model_size_bytes
        self.last_comm = download_bytes + upload_bytes
        
        # Store detailed communication metrics
        self.communication_metrics = {
            'download_bytes': download_bytes,
            'upload_bytes': upload_bytes,
            'total_bytes': self.last_comm,
            'compression_ratio': 1.0,  # No compression in dense case
            'model_size_bytes': model_size_bytes
        }
        
        return total_loss / len(self.train_loader), 100.0 * correct / total

    def get_metrics(self):
        """Get detailed metrics including communication costs."""
        return {
            'flops': self.last_flops,
            'memory': self.last_mem,
            'communication': self.last_comm,
            'sparsity': 0.0,  # No sparsity in dense case
            'download_bytes': self.communication_metrics['download_bytes'],
            'upload_bytes': self.communication_metrics['upload_bytes'],
            'compression_ratio': 1.0,
            'model_size_bytes': self.communication_metrics['model_size_bytes']
        }

    def local_evaluate(self) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
