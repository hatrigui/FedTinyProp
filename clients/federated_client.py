import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import flwr as fl
from models.config import get_tinyprop_config
from utils.adaptive_sparsification import AdaptiveSparsifier
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from utils.training_helpers import compute_adaptive_ratio, compute_sparsity_and_flops
from typing import Dict, Tuple
import os
import psutil
import pandas as pd
from torch.utils.data import DataLoader
from utils.flops_calculator import compute_model_flops
from utils.memory_calculator import estimate_model_memory

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, model: nn.Module, train_loader: DataLoader, 
                 test_loader: DataLoader, cfg: dict, device: str = None, dataset_name: str = None):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dataset_name = dataset_name
        
        self.metrics = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'grad_norm': [],
            'phi_k': [],
            'skipped_batches': [],
            'memory_usage': [],
            'communication_cost': [],
            'weight_deltas': [],
            'compression_ratio': []
        }
        
        self.initial_weights = None
        self.global_weights = None
        
        self.last_flops = 0.0
        self.last_mem = 0.0
        self.last_mem_saved = 0.0
        self.last_comm = 0.0
        self.last_sparsity = 0.0
        self.last_avg_grad_norm = 0.0
        self.last_phi = 0.0
        self.num_skipped_batches = 0
        self.total_batches = 0
        self.compression_ratio = 1.0
        self.layer_sparsity = {}
        
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.cfg["optimizer"]["lr"],
            momentum=self.cfg["optimizer"].get("momentum", 0.9),
            weight_decay=self.cfg["optimizer"].get("weight_decay", 0.0),
            nesterov=self.cfg["optimizer"].get("nesterov", False)
        )
        
        if "lr_scheduler" in self.cfg:
            scheduler_cfg = self.cfg["lr_scheduler"]
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_cfg["T_max"],
                eta_min=scheduler_cfg["eta_min"]
            )
        else:
            self.scheduler = None
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.get("label_smoothing", 0.0))
        
        tinyprop_params = self.cfg["tinyprop_params"]
        self.S_min = tinyprop_params.S_min
        self.S_max = tinyprop_params.S_max
        self.zeta = tinyprop_params.zeta
        self.skip_threshold = self.cfg.get("skip_threshold", 2.5)
        self.phi_min = self.cfg.get("phi_min", 0.2)
        
        self.sparsifier = AdaptiveSparsifier(
            initial_sparsity=self.S_min,
            target_sparsity=self.S_max,
            total_rounds=self.cfg.get("total_rounds", 100),
            energy_budget=self.cfg.get("energy_budget", None)
        )
        
        self.weight_deltas = {}
        self.initial_grad_norm = None
        self.smoothed_phi = None
        self.initial_grad_norms = []
        self.INITIAL_GRAD_NORM_BATCHES = 5
        self.adaptive_sparsity = True
        
        self.phi_ema_alpha = 0.9
        self.target_sparsity_adjustment_rate = 0.1
        
        self.metrics = {
            "nonzero_gradients": [],
            "batch_flops": [],
            "peak_memory": 0,
            "communication_cost": 0,
            "effective_sparsity": [],
            "phi_values": [],
            "skipped_updates": 0
        }
        
        if not hasattr(self.model.tpLayer, 'stats'):
            self.model.tpLayer.stats = {
                "skipped_batches": 0,
                "total_batches": 0,
                "phi_k_history": [],
                "loss_change_history": [],
                "loss_history": [],
                "last_batch_loss": None,
                "loss_threshold": 0.01
            }
        
        quantization_cfg = cfg.get("quantization", {"bits": 32, "enabled": False})
        self.quantization_bits = quantization_cfg.get("bits", 32)
        self.quantization_enabled = quantization_cfg.get("enabled", False)
        self.quantization_error = 0.0
        self.avg_scale_factor = 1.0
        self.quantization_metrics = {
            "errors": [],
            "scale_factors": []
        }

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), parameters):
            if isinstance(param, np.ndarray):
                param = torch.from_numpy(param)
            state_dict[key] = param.to(self.device)
        self.model.load_state_dict(state_dict)

    def get_metrics(self):
        return {
            'flops': self.last_flops,
            'memory': self.last_mem,
            'communication': self.last_comm,
            'sparsity': self.last_sparsity,
            'grad_norm': self.last_avg_grad_norm,
            'phi': self.last_phi,
            'skipped_batches': self.num_skipped_batches,
            'compression_ratio': self.compression_ratio,
            'layer_sparsity': self.layer_sparsity
        }

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        self.set_parameters(parameters)
        batch_size = config.get("batch_size", 32) if config else 32
        local_epochs = config.get("local_epochs", 1) if config else 1
        
        loss, accuracy = self.train(num_epochs=local_epochs, batch_size=batch_size)
        updated_parameters = self.get_parameters()
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "phi": float(self.last_phi),
            "skipped_batches": self.num_skipped_batches,
            "sparsity": self.last_sparsity
        }
        
        return updated_parameters, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset."""
        self.set_parameters(parameters)
        if self.test_loader is not None:
            acc = self.local_evaluate(self.test_loader)
            return float(acc), len(self.test_loader.dataset)  
        return 0.0, 0  

    def compute_grad_norm(self) -> float:
        """Compute the L2 norm of all gradients in the model."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def update_adaptive_sparsity(self, current_grad_norm: float) -> float:
        """Update the adaptive sparsity coefficient with EMA smoothing."""
        epsilon = 1e-8
        if self.initial_grad_norm is None:
            if abs(current_grad_norm) < epsilon:
                self.initial_grad_norm = epsilon
            else:
                self.initial_grad_norm = current_grad_norm
            self.smoothed_phi = 1.0
            return 1.0
        
        raw_phi = compute_adaptive_ratio(current_grad_norm, self.initial_grad_norm, self.phi_min)
        
        if self.smoothed_phi is None:
            self.smoothed_phi = raw_phi
        else:
            self.smoothed_phi = self.phi_ema_alpha * self.smoothed_phi + (1 - self.phi_ema_alpha) * raw_phi
        
        self.last_phi = self.smoothed_phi
        return self.smoothed_phi

    def should_skip_update(self, current_grad_norm: float) -> bool:
        """Determine if the current update should be skipped based on gradient norm."""
        return current_grad_norm < self.skip_threshold

    def apply_gradient_sparsification(self, phi: float) -> None:
        """Apply gradient sparsification with top-k selection."""
        total_nonzero = 0
        total_elements = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layer_sparsity = max(self.S_min, min(self.S_max, self.zeta * phi))
                
                # Calculate number of elements to keep
                num_elements = param.grad.numel()
                num_keep = int(num_elements * (1 - layer_sparsity))
                
                # Get top-k values and indices
                grad_flat = param.grad.data.view(-1)
                values, indices = torch.topk(torch.abs(grad_flat), k=num_keep)
                
                # Create sparse gradient
                sparse_grad = torch.zeros_like(grad_flat)
                sparse_grad[indices] = grad_flat[indices]
                
                # Update gradient
                param.grad.data = sparse_grad.view(param.grad.data.shape)
                
                nonzero_count = num_keep
                total_nonzero += nonzero_count
                total_elements += num_elements
                
                self.layer_sparsity[name] = 1.0 - (nonzero_count / num_elements)
        
        if total_elements > 0:
            effective_sparsity = 1.0 - (total_nonzero / total_elements)
            self.metrics["nonzero_gradients"].append(total_nonzero)
            self.metrics["effective_sparsity"].append(effective_sparsity)

    def compute_metrics(self) -> Dict[str, float]:
        metrics = {
            "flops": self.last_flops,
            "memory": 0.0,
            "memory_saved": 0.0,
            "communication": 0.0,
            "sparsity": 0.0,
            "layer_flops": {}
        }

        sample_input, _ = next(iter(self.train_loader))
        single_sample_shape = (1, *sample_input.shape[1:])
        _, layer_flops = compute_model_flops(
            self.model,
            single_sample_shape,
            layer_sparsity_dict=self.layer_sparsity
        )
        metrics["layer_flops"] = {k: v * sample_input.shape[0] for k, v in layer_flops.items()}
        metrics["sparsity"] = np.mean(list(self.layer_sparsity.values()))

        input_shape = sample_input.shape[1:]
        mem_report = estimate_model_memory(self.model, batch_size=sample_input.shape[0], input_shape=input_shape)
        batch_memory = mem_report["total_MB"]
        total_batches = len(self.train_loader)
        processed_batches = total_batches - self.num_skipped_batches
        skipped_batches = self.num_skipped_batches

        # Calculate memory used for processed batches
        used_memory = processed_batches * batch_memory
        # Calculate memory saved (difference between total and used)
        saved_memory = skipped_batches * batch_memory

        metrics["memory"] = used_memory
        metrics["memory_saved"] = saved_memory
        self.last_mem = metrics["memory"]
        self.last_mem_saved = metrics["memory_saved"]

        total_bytes = sum(
            indices.numel() * indices.element_size() + values.numel() * values.element_size()
            for indices, values in self.weight_deltas.values()
        )
        metrics["communication"] = total_bytes
        original_size = sum(p.numel() * 4 for p in self.model.parameters() if p.requires_grad)
        metrics["compression_ratio"] = total_bytes / original_size if original_size else 1.0

        return metrics


    def get_quantization_metrics(self) -> Tuple[float, float]:
        """Return the current quantization error and average scale factor."""
        if not self.quantization_enabled:
            return 0.0, 1.0
        if self.quantization_metrics["errors"]:
            return np.mean(self.quantization_metrics["errors"]), np.mean(self.quantization_metrics["scale_factors"])
        return 0.0, 1.0

    def apply_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Apply quantization to a tensor and return the quantized tensor, error, and scale factor."""
        if not self.quantization_enabled or self.quantization_bits == 32:
            return tensor, 0.0, 1.0
            
        max_val = tensor.abs().max().item()
        scale_factor = (2 ** (self.quantization_bits - 1) - 1) / max_val if max_val > 0 else 1.0
        quantized = torch.round(tensor * scale_factor) / scale_factor
        
        error = torch.mean((tensor - quantized).abs()).item()
        self.quantization_metrics["errors"].append(error)
        self.quantization_metrics["scale_factors"].append(scale_factor)
        
        return quantized, error, scale_factor

    def train(self, num_epochs: int = 1, batch_size: int = 32) -> Tuple[float, float]:
        """Train the model on the local dataset."""
        print(f"\n[Client Debug] Starting training for {num_epochs} epochs with batch size {batch_size}")
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self.num_skipped_batches = 0
        
        initial_weights = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        epoch_grad_norms = []
        total_flops = 0
        total_memory = 0
        total_communication = 0
        epoch_effective_sparsities = []
        
        current_round = getattr(self.model, 'current_round', 0)
        total_rounds = self.cfg.get("total_rounds", 100)
        
        self.model.tpLayer.current_round = current_round
        self.model.tpLayer.adjust_loss_threshold(current_round, total_rounds)
        
        progress = current_round / total_rounds
        # Make sparsity increase more aggressively by using a quadratic progression
        progressive_factor = 1.0 + (self.S_max - self.S_min) * (progress ** 2)
        
        # Add early sparsity boost
        if current_round < total_rounds * 0.3:  # First 30% of rounds
            progressive_factor *= 1.5  # 50% boost in early rounds

        max_grad_norm = 1.0  
        grad_clip_threshold = 5.0
        
        # Track loss history for zeta adjustment
        loss_history = []
        loss_window_size = 5
        zeta_adjustment_threshold = 0.01  # Minimum loss change to trigger zeta adjustment
        
        for epoch in range(num_epochs):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track loss history
                loss_history.append(loss.item())
                if len(loss_history) > loss_window_size:
                    loss_history.pop(0)
                
                # Adjust zeta based on loss plateauing
                if len(loss_history) == loss_window_size:
                    loss_change = abs(loss_history[-1] - loss_history[0])
                    if loss_change < zeta_adjustment_threshold:
                        # Loss is plateauing, increase zeta to maintain sparsity
                        self.zeta *= 1.05  # 5% increase
                    elif loss_change > zeta_adjustment_threshold * 2:
                        # Loss is changing significantly, decrease zeta to allow more updates
                        self.zeta *= 0.95  # 5% decrease
                    # Keep zeta within reasonable bounds
                    self.zeta = max(0.1, min(2.0, self.zeta))
                
                loss.backward()
                if self.quantization_enabled:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            quantized_grad, error, scale = self.apply_quantization(param.grad)
                            param.grad.data = quantized_grad
                
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # Apply gradient clipping
                if grad_norm > grad_clip_threshold:
                    scale = grad_clip_threshold / (grad_norm + 1e-6)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data *= scale
                
                if grad_norm > 0:
                    epoch_grad_norms.append(grad_norm)
                    
                    self.model.tpLayer.update_phi(self.model.tpParams, loss.item(), batch_idx)
                    
                    if self.model.tpLayer.should_skip_batch(loss.item(), self.model.tpParams):
                        self.num_skipped_batches += 1
                        continue
                    if self.adaptive_sparsity:
                        phi = self.update_adaptive_sparsity(grad_norm)
                        
                        self.apply_gradient_sparsification(phi)
                        
                        base_sparsity = self.S_min + (self.S_max - self.S_min) * (1 - phi)
                        local_sparsity = min(self.S_max, base_sparsity * progressive_factor)
                        epoch_effective_sparsities.append(local_sparsity)
                        
                        # Calculate FLOPs for the entire batch at once
                        batch_flops, _ = compute_model_flops(self.model, inputs.shape, self.layer_sparsity)
                        total_flops += batch_flops
                        
                        # Debug logging for first batch
                        if batch_idx == 0:
                            per_sample_flops = batch_flops / inputs.shape[0]
                            print(f"[Debug] Per-sample FLOPs: {per_sample_flops:.2f}")
                            print(f"[Debug] Batch size: {inputs.shape[0]}")
                            print(f"[Debug] Total batch FLOPs: {batch_flops:.2f}")
                
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                process = psutil.Process(os.getpid())
                total_memory = max(total_memory, process.memory_info().rss / 1024 / 1024)
                
                if batch_idx % 10 == 0:
                    print(f"\n[Batch {batch_idx}] Debug Info:")
                    print(f"  - Loss: {loss.item():.4f}")
                    print(f"  - Grad Norm: {grad_norm:.4f}")
                    print(f"  - Current phi_k: {self.model.phi_k:.4f}")
                    print(f"  - Current zeta: {self.zeta:.4f}")
                    print(f"  - Skipped batches: {self.num_skipped_batches}")
                    print(f"  - Memory usage: {total_memory:.2f} MB")
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.last_flops = total_flops
        self.last_mem = total_memory
        self.last_comm = total_communication
        self.last_sparsity = np.mean(epoch_effective_sparsities) if epoch_effective_sparsities else 0.0
        self.last_avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        self.last_phi = self.model.phi_k
        
        self.weight_deltas = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial = initial_weights[name]
                final = param.data
                delta = final - initial
                if delta.numel() > 0:
                    mask = delta.abs() > 1e-6  
                    if len(delta.shape) > 1:
                        flat_delta = delta.view(-1)
                        flat_mask = mask.view(-1)
                        indices = flat_mask.nonzero().squeeze()
                        values = flat_delta[indices]
                    else:
                        indices = mask.nonzero().squeeze()
                        values = delta[indices]
                    
                    if indices.numel() > 0:  
                        self.weight_deltas[name] = (indices, values)
        
        return total_loss / len(self.train_loader), 100. * correct / total

    def local_evaluate(self, data_loader):
        """Evaluate model on local data."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def _save_metrics_to_csv(self):
        """Save client metrics to CSV file."""
        df = pd.DataFrame(self.metrics)
        df.to_csv(f"client_{self.client_id}_metrics.csv", index=False)