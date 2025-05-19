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
        super().__init__()
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dataset_name = dataset_name
        
        # Check if we're using dense baseline
        self.is_dense_baseline = (
            self.cfg["tinyprop_params"].S_min == 0.0 and 
            self.cfg["tinyprop_params"].S_max == 0.0 and 
            self.cfg["tinyprop_params"].zeta == 0.0 and
            self.cfg.get("skip_threshold", float("inf")) == float("inf")
        )
        
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
        self.skip_threshold = self.cfg.get("skip_threshold", float("inf"))
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
        
        quantization_cfg = cfg.get("quantization", {
            "bits": 8,
            "enabled": False,
            "adaptive": True,
            "min_bits": 4,
            "max_bits": 16,
            "layer_specific": True,
            "error_threshold": 0.01,
            "momentum": 0.9
        })
        self.quantization_bits = quantization_cfg.get("bits", 8)
        self.quantization_enabled = quantization_cfg.get("enabled", False)
        self.adaptive_quantization = quantization_cfg.get("adaptive", True)
        self.min_bits = quantization_cfg.get("min_bits", 4)
        self.max_bits = quantization_cfg.get("max_bits", 16)
        self.layer_specific = quantization_cfg.get("layer_specific", True)
        self.error_threshold = quantization_cfg.get("error_threshold", 0.01)
        self.quantization_momentum = quantization_cfg.get("momentum", 0.9)
        
        self.quantization_error = 0.0
        self.avg_scale_factor = 1.0
        self.layer_quantization_stats = {}
        self.quantization_metrics = {
            "errors": [],
            "scale_factors": [],
            "layer_bits": {},
            "layer_errors": {},
            "layer_scale_factors": {}
        }

        self.communication_metrics = {
            'download_bytes': 0.0,
            'upload_bytes': 0.0,
            'total_bytes': 0.0,
            'compression_ratio': 1.0,
            'model_size_bytes': 0.0,
            'layer_communications': {}
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

    def get_metrics(self) -> Dict[str, float]:
        """Get client metrics."""
        return {
            'flops': self.last_flops,
            'memory': self.last_mem,
            'memory_saved': self.last_mem_saved,
            'communication': self.last_comm,
            'sparsity': self.last_sparsity,
            'skipped_batches': self.num_skipped_batches,
            'download_bytes': self.communication_metrics['download_bytes'],
            'upload_bytes': self.communication_metrics['upload_bytes'],
            'model_size_bytes': self.communication_metrics['model_size_bytes'],
            'compression_ratio': self.communication_metrics['compression_ratio']
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
        if self.is_dense_baseline:
            return False
        return current_grad_norm < self.skip_threshold

    def apply_gradient_sparsification(self, phi: float) -> None:
        """Apply gradient sparsification with top-k selection."""
        if self.is_dense_baseline:
            return  # Skip sparsification for dense baseline
            
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
            "layer_flops": {},
            "download_bytes": 0.0,
            "upload_bytes": 0.0,
            "compression_ratio": 1.0,
            "model_size_bytes": 0.0
        }

        # Calculate model size and memory metrics
        total_params = 0
        total_nonzero = 0
        total_memory = 0
        total_memory_saved = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_size = param.numel()
                total_params += param_size
                
                # Calculate memory for dense parameter (4 bytes per float32)
                dense_memory = param_size * 4
                total_memory += dense_memory
                
                # Calculate memory for sparse parameter if applicable
                if name in self.weight_deltas:
                    update_data = self.weight_deltas[name]
                    if isinstance(update_data, tuple):
                        indices, values = update_data
                        if isinstance(indices, torch.Tensor) and isinstance(values, torch.Tensor):
                            # Calculate sparse memory (indices + values + overhead)
                            max_index = param_size - 1
                            if max_index <= 255:  # uint8
                                index_bytes = 1
                            elif max_index <= 65535:  # uint16
                                index_bytes = 2
                            else:  # uint32
                                index_bytes = 4
                            
                            sparse_memory = (indices.numel() * index_bytes +  # indices
                                           values.numel() * 4 +              # values (float32)
                                           3)                               # format overhead
                            
                            # Only use sparse if it's more efficient
                            if sparse_memory < dense_memory:
                                total_memory_saved += (dense_memory - sparse_memory)
                                total_nonzero += indices.numel()
                            else:
                                total_nonzero += param_size
                        else:
                            total_nonzero += param_size
                    else:
                        total_nonzero += param_size
                else:
                    total_nonzero += param_size

        # Calculate model size for download cost (full model)
        model_size_bytes = total_memory
        metrics["model_size_bytes"] = model_size_bytes
        metrics["download_bytes"] = model_size_bytes  # Each client downloads full model
        metrics["memory"] = total_memory / (1024 * 1024)  # Convert to MB
        metrics["memory_saved"] = total_memory_saved / (1024 * 1024)  # Convert to MB

        # Calculate sparse upload cost with optimized format
        upload_bytes = 0
        layer_communications = {}
        
        # Higher threshold for significant updates
        SIGNIFICANT_THRESHOLD = 1e-4  # Increased from 1e-6
        
        for name, update_data in self.weight_deltas.items():
            if isinstance(update_data, tuple):
                indices, values = update_data
                if not isinstance(indices, torch.Tensor) or not isinstance(values, torch.Tensor):
                    continue
                    
                param = self.model.state_dict()[name]
                
                # Only keep significant updates
                mask = values.abs() > SIGNIFICANT_THRESHOLD
                indices = indices[mask]
                values = values[mask]
                
                if indices.numel() == 0:
                    continue
                
                # Optimize index storage
                max_index = param.numel() - 1
                if max_index <= 255:  # Can use uint8
                    indices = indices.to(torch.uint8)
                    index_bytes = 1
                elif max_index <= 65535:  # Can use uint16
                    indices = indices.to(torch.uint16)
                    index_bytes = 2
                else:  # Must use uint32
                    indices = indices.to(torch.uint32)
                    index_bytes = 4
                
                # Calculate bytes needed for this layer's update
                indices_bytes = indices.numel() * index_bytes
                values_bytes = values.numel() * 4  # float32 = 4 bytes
                format_overhead = 3  # Reduced format overhead
                
                # Calculate total bytes for this layer
                layer_comm = indices_bytes + values_bytes + format_overhead
                
                # Only use sparse format if it's more efficient than dense
                dense_layer_bytes = param.numel() * 4
                if layer_comm >= dense_layer_bytes:
                    # Fall back to dense format
                    layer_comm = dense_layer_bytes
                    self.weight_deltas[name] = (None, param.data.cpu())
                
                upload_bytes += layer_comm
                layer_communications[name] = layer_comm
            else:
                # Handle dense update
                param = self.model.state_dict()[name]
                upload_bytes += param.numel() * 4
                layer_communications[name] = param.numel() * 4

        metrics["upload_bytes"] = upload_bytes
        metrics["communication"] = upload_bytes + model_size_bytes  # Total = upload + download
        metrics["layer_communication"] = layer_communications
        
        # Calculate effective compression ratio
        if upload_bytes > 0:
            metrics["compression_ratio"] = model_size_bytes / upload_bytes
        else:
            metrics["compression_ratio"] = 1.0
            
        # Calculate effective sparsity
        if total_params > 0:
            metrics["sparsity"] = 1.0 - (total_nonzero / total_params)
        
        # Store metrics for logging
        self.last_comm = metrics["communication"]
        self.last_mem = metrics["memory"]
        self.last_mem_saved = metrics["memory_saved"]
        self.communication_metrics = {
            'download_bytes': metrics["download_bytes"],
            'upload_bytes': metrics["upload_bytes"],
            'total_bytes': metrics["communication"],
            'compression_ratio': metrics["compression_ratio"],
            'model_size_bytes': metrics["model_size_bytes"],
            'layer_communications': layer_communications
        }
        
        return metrics

    def get_quantization_metrics(self) -> Tuple[float, float]:
        """Return the current quantization error and average scale factor."""
        if not self.quantization_enabled:
            return 0.0, 1.0
        if self.quantization_metrics["errors"]:
            return np.mean(self.quantization_metrics["errors"]), np.mean(self.quantization_metrics["scale_factors"])
        return 0.0, 1.0

    def get_layer_optimal_bits(self, tensor: torch.Tensor, layer_name: str) -> int:
        """Determine optimal bit-width for a layer based on its statistics."""
        if not self.adaptive_quantization or not self.layer_specific:
            return self.quantization_bits
            
        if layer_name not in self.layer_quantization_stats:
            self.layer_quantization_stats[layer_name] = {
                "max_val": 0.0,
                "min_val": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "error": 0.0,
                "bits": self.quantization_bits
            }
            
        stats = self.layer_quantization_stats[layer_name]
        
        # Update statistics with momentum
        current_max = tensor.abs().max().item()
        current_min = tensor.abs().min().item()
        current_mean = tensor.abs().mean().item()
        current_std = tensor.abs().std().item()
        
        stats["max_val"] = self.quantization_momentum * stats["max_val"] + (1 - self.quantization_momentum) * current_max
        stats["min_val"] = self.quantization_momentum * stats["min_val"] + (1 - self.quantization_momentum) * current_min
        stats["mean"] = self.quantization_momentum * stats["mean"] + (1 - self.quantization_momentum) * current_mean
        stats["std"] = self.quantization_momentum * stats["std"] + (1 - self.quantization_momentum) * current_std
        
        # Calculate dynamic range
        dynamic_range = stats["max_val"] - stats["min_val"]
        if dynamic_range < 1e-6:
            return self.min_bits
            
        # Calculate signal-to-noise ratio (SNR)
        snr = 20 * np.log10(stats["mean"] / (stats["std"] + 1e-6))
        
        # Adjust bits based on SNR and error threshold
        if snr > 40:  # High SNR, can use fewer bits
            target_bits = max(self.min_bits, int(self.quantization_bits * 0.75))
        elif snr < 20:  # Low SNR, need more bits
            target_bits = min(self.max_bits, int(self.quantization_bits * 1.25))
        else:
            target_bits = self.quantization_bits
            
        # Adjust based on previous error
        if stats["error"] > self.error_threshold:
            target_bits = min(self.max_bits, target_bits + 1)
        elif stats["error"] < self.error_threshold * 0.5:
            target_bits = max(self.min_bits, target_bits - 1)
            
        return target_bits

    def apply_quantization(self, tensor: torch.Tensor, layer_name: str = None) -> Tuple[torch.Tensor, float, float]:
        """Apply adaptive quantization to a tensor and return the quantized tensor, error, and scale factor."""
        if not self.quantization_enabled:
            return tensor, 0.0, 1.0
            
        # Determine optimal bit-width for this layer
        bits = self.get_layer_optimal_bits(tensor, layer_name) if layer_name else self.quantization_bits
        
        # Calculate scale factor based on dynamic range
        max_val = tensor.abs().max().item()
        if max_val < 1e-6:
            return tensor, 0.0, 1.0
            
        scale_factor = (2 ** (bits - 1) - 1) / max_val
        quantized = torch.round(tensor * scale_factor) / scale_factor
        
        # Calculate quantization error
        error = torch.mean((tensor - quantized).abs()).item()
        
        # Update layer statistics
        if layer_name:
            if layer_name not in self.layer_quantization_stats:
                self.layer_quantization_stats[layer_name] = {
                    "max_val": 0.0,
                    "min_val": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "error": 0.0,
                    "bits": bits
                }
            self.layer_quantization_stats[layer_name]["error"] = error
            self.layer_quantization_stats[layer_name]["bits"] = bits
            
            # Update layer metrics
            if layer_name not in self.quantization_metrics["layer_errors"]:
                self.quantization_metrics["layer_errors"][layer_name] = []
                self.quantization_metrics["layer_scale_factors"][layer_name] = []
                self.quantization_metrics["layer_bits"][layer_name] = []
            
            self.quantization_metrics["layer_errors"][layer_name].append(error)
            self.quantization_metrics["layer_scale_factors"][layer_name].append(scale_factor)
            self.quantization_metrics["layer_bits"][layer_name].append(bits)
        
        # Update global metrics
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
        
        # Get initial memory estimate
        sample_input, _ = next(iter(self.train_loader))
        mem_report = estimate_model_memory(self.model, batch_size=sample_input.shape[0], input_shape=sample_input.shape[1:])
        total_memory = mem_report["total_MB"]
        
        # Initialize layer sparsity for FLOPs calculation
        layer_sparsity = {name: 0.0 for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
        
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
                        self.zeta *= 1.05
                    elif loss_change > zeta_adjustment_threshold * 2:
                        self.zeta *= 0.95
                    self.zeta = max(0.1, min(2.0, self.zeta))
                
                loss.backward()
                if self.quantization_enabled:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            quantized_grad, error, scale = self.apply_quantization(param.grad, name)
                            param.grad.data = quantized_grad
                
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                if grad_norm > grad_clip_threshold:
                    scale = grad_clip_threshold / (grad_norm + 1e-6)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data *= scale
                
                if grad_norm > 0:
                    epoch_grad_norms.append(grad_norm)
                    
                    # Skip batch skipping logic for dense baseline
                    if not self.is_dense_baseline:
                        self.model.tpLayer.update_phi(self.model.tpParams, loss.item(), batch_idx)
                        
                        if self.model.tpLayer.should_skip_batch(loss.item(), self.model.tpParams):
                            self.num_skipped_batches += 1
                            continue
                    
                    # Calculate FLOPs only for non-skipped batches
                    if self.is_dense_baseline:
                        # For dense baseline, use zero sparsity
                        layer_sparsity = {name: 0.0 for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
                    else:
                        if self.adaptive_sparsity:
                            phi = self.update_adaptive_sparsity(grad_norm)
                            self.apply_gradient_sparsification(phi)
                            base_sparsity = self.S_min + (self.S_max - self.S_min) * (1 - phi)
                            local_sparsity = min(self.S_max, base_sparsity * progressive_factor)
                            epoch_effective_sparsities.append(local_sparsity)
                            # Update layer sparsity for FLOPs calculation
                            layer_sparsity = {name: local_sparsity for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
                    
                    # Calculate FLOPs for the current batch
                    batch_flops, _ = compute_model_flops(self.model, inputs.shape, layer_sparsity)
                    total_flops += batch_flops
                    

                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.last_flops = total_flops
        self.last_mem = total_memory
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
                    # Use higher threshold for significant updates
                    mask = delta.abs() > 1e-4  # Increased from 1e-6
                    
                    if len(delta.shape) > 1:
                        flat_delta = delta.view(-1)
                        flat_mask = mask.view(-1)
                        indices = flat_mask.nonzero().squeeze()
                        values = flat_delta[indices]
                    else:
                        indices = mask.nonzero().squeeze()
                        values = delta[indices]
                    
                    if indices.numel() > 0:
                        # Only store if sparse format would be more efficient
                        param_size = param.numel() * 4  # dense size in bytes
                        sparse_size = (indices.numel() * (2 if param.numel() <= 65535 else 4) + 
                                     values.numel() * 4 + 3)  # sparse size with overhead
                        
                        if sparse_size < param_size:
                            self.weight_deltas[name] = (indices, values)
                        else:
                            self.weight_deltas[name] = (None, param.data.cpu())  # Store full parameter
        
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