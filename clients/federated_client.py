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
        
        # Add FedPrune flag
        self.is_fedprune = self.cfg.get("use_fedprune", False)
        
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
        """Get model parameters as a list of NumPy arrays."""
        state_dict = self.model.state_dict()
        parameters = []
        for val in state_dict.values():
            if isinstance(val, torch.Tensor):
                # Convert to float32 before converting to numpy
                if val.dtype == torch.long:
                    val = val.float()
                parameters.append(val.cpu().numpy())
            else:
                parameters.append(val)
        return parameters

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
            'compression_ratio': self.communication_metrics['compression_ratio'],
            'smoothed_adaptivity_factor': self.smoothed_phi if hasattr(self, 'smoothed_phi') else None
        }

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        self.set_parameters(parameters)
        batch_size = config.get("batch_size", 32) if config else 32
        local_epochs = config.get("local_epochs", 1) if config else 1
        
        # Convert parameters to global_params dict for FedProx
        global_params = {name: torch.tensor(param) for name, param in zip(self.model.state_dict().keys(), parameters)}
        
        loss, accuracy = self.train(
            num_epochs=local_epochs, 
            batch_size=batch_size,
            global_params=global_params if self.cfg.get("use_fedprox", False) else None
        )
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
        if self.is_dense_baseline or self.is_fedprune:
            return False
        return current_grad_norm < self.skip_threshold

    def apply_gradient_sparsification(self, phi: float) -> None:
        """Apply gradient sparsification with top-k selection."""
        if self.is_dense_baseline or self.is_fedprune:
            return  # Skip sparsification for dense baseline or FedPrune
            
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
        """Compute and return various metrics about the model and training."""
        metrics = {}
        
        # Calculate model size and memory usage
        total_params = 0
        nonzero_params = 0
        total_memory = 0
        sparse_memory = 0
        
        # Track layer-wise sparsity for FLOPs calculation
        layer_sparsity = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Count total parameters
                total_params += param.numel()
                
                # Count nonzero parameters
                nonzero_mask = param.data != 0
                nonzero_count = nonzero_mask.sum().item()
                nonzero_params += nonzero_count
                
                # Calculate memory usage
                param_memory = param.numel() * 4  # 4 bytes per parameter (float32)
                total_memory += param_memory
                
                # Calculate sparse memory (only nonzero parameters)
                sparse_memory += nonzero_count * 4
                
                # Store layer sparsity for FLOPs calculation
                if len(param.shape) > 1:  # Only for weight matrices
                    layer_sparsity[name] = 1.0 - (nonzero_count / param.numel())
        
        # Calculate memory saved
        self.last_mem = total_memory
        self.last_mem_saved = max(0, total_memory - sparse_memory)
        
        # Calculate communication metrics
        self.communication_metrics['model_size_bytes'] = total_memory
        self.communication_metrics['download_bytes'] = total_memory  # Full model download
        self.communication_metrics['upload_bytes'] = sparse_memory  # Only nonzero parameters
        
        # Calculate compression ratio
        if sparse_memory > 0:
            self.communication_metrics['compression_ratio'] = total_memory / sparse_memory
        else:
            self.communication_metrics['compression_ratio'] = 1.0
            
        # Calculate total communication
        self.last_comm = self.communication_metrics['download_bytes'] + self.communication_metrics['upload_bytes']
        
        # Calculate sparsity
        self.last_sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
        
        # Calculate FLOPs with proper sparsity consideration
        if self.is_fedprune:
            # For FedPrune, use the actual layer-wise sparsity from the masks
            self.last_flops = compute_model_flops(self.model, layer_sparsity=layer_sparsity)
        else:
            # For other methods, use the existing FLOPs calculation
            self.last_flops = compute_model_flops(self.model)
        
        return {
            'flops': self.last_flops,
            'memory': self.last_mem,
            'memory_saved': self.last_mem_saved,
            'communication': self.last_comm,
            'sparsity': self.last_sparsity,
            'download_bytes': self.communication_metrics['download_bytes'],
            'upload_bytes': self.communication_metrics['upload_bytes'],
            'model_size_bytes': self.communication_metrics['model_size_bytes'],
            'compression_ratio': self.communication_metrics['compression_ratio']
        }

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

    def train(self, num_epochs: int = 1, batch_size: int = 32, global_params=None) -> Tuple[float, float]:
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
        
        # Skip TinyProp layer updates for FedPrune
        if not self.is_fedprune:
            self.model.tpLayer.current_round = current_round
            self.model.tpLayer.adjust_loss_threshold(current_round, total_rounds)
        
        progress = current_round / total_rounds
        progressive_factor = 1.0 + (self.S_max - self.S_min) * (progress ** 2)
        
        if current_round < total_rounds * 0.3:
            progressive_factor *= 1.5

        max_grad_norm = 1.0  
        grad_clip_threshold = 5.0
        
        loss_history = []
        loss_window_size = 5
        zeta_adjustment_threshold = 0.01
        
        sample_input, _ = next(iter(self.train_loader))
        mem_report = estimate_model_memory(self.model, batch_size=sample_input.shape[0], input_shape=sample_input.shape[1:])
        total_memory = mem_report["total_MB"]
        
        # Initialize layer sparsity tracking
        layer_sparsity = {}
        if self.is_fedprune:
            # For FedPrune, calculate actual layer sparsity from masks
            for name, param in self.model.named_parameters():
                if len(param.shape) > 1:  # Only for weight matrices
                    nonzero_count = (param.data != 0).sum().item()
                    layer_sparsity[name] = 1.0 - (nonzero_count / param.numel())
            # Add to epoch sparsities for tracking
            epoch_effective_sparsities.append(np.mean(list(layer_sparsity.values())))
        else:
            layer_sparsity = {name: 0.0 for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
        
        for epoch in range(num_epochs):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # FedProx proximal term with proper parameter handling
                if self.cfg.get("use_fedprox", False) and global_params is not None:
                    fedprox_mu = self.cfg.get("fedprox_mu", 0.1)
                    proximal_term = 0.0
                    
                    # Only iterate over trainable parameters
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            global_param = global_params[name].to(self.device)
                            diff = param - global_param
                            proximal_term += (diff**2).sum()
                    
                    loss += (fedprox_mu / 2) * proximal_term
                
                loss_history.append(loss.item())
                if len(loss_history) > loss_window_size:
                    loss_history.pop(0)
                
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
                    
                    # Skip TinyProp updates for FedPrune
                    if not self.is_dense_baseline and not self.is_fedprune:
                        self.model.tpLayer.update_phi(self.model.tpParams, loss.item(), batch_idx)
                        
                        if self.model.tpLayer.should_skip_batch(loss.item(), self.model.tpParams):
                            self.num_skipped_batches += 1
                            continue
                    
                    if self.is_dense_baseline:
                        layer_sparsity = {name: 0.0 for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
                    elif self.is_fedprune:
                        # For FedPrune, recalculate layer sparsity from current masks
                        for name, param in self.model.named_parameters():
                            if len(param.shape) > 1:  # Only for weight matrices
                                nonzero_count = (param.data != 0).sum().item()
                                layer_sparsity[name] = 1.0 - (nonzero_count / param.numel())
                        # Add to epoch sparsities for tracking
                        epoch_effective_sparsities.append(np.mean(list(layer_sparsity.values())))
                    else:
                        if self.adaptive_sparsity:
                            phi = self.update_adaptive_sparsity(grad_norm)
                            self.apply_gradient_sparsification(phi)
                            base_sparsity = self.S_min + (self.S_max - self.S_min) * (1 - phi)
                            local_sparsity = min(self.S_max, base_sparsity * progressive_factor)
                            epoch_effective_sparsities.append(local_sparsity)
                            layer_sparsity = {name: local_sparsity for name, _ in self.model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
                    
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
        
        # Calculate weight deltas and compression metrics
        self.weight_deltas = {}
        total_dense_size = 0
        total_sparse_size = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial = initial_weights[name]
                final = param.data
                delta = final - initial
                
                if delta.numel() > 0:
                    # Calculate relative threshold based on layer's dynamic range
                    max_abs_value = delta.abs().max().item()
                    min_abs_value = delta[delta != 0].abs().min().item() if (delta != 0).any() else max_abs_value
                    dynamic_range = max_abs_value - min_abs_value
                    
                    current_round = getattr(self.model, 'current_round', 0)
                    total_rounds = self.cfg.get("total_rounds", 100)
                    progress = current_round / total_rounds
                    
                    base_threshold = 1e-3 * (1 - 0.9 * progress)
                    relative_threshold = base_threshold * max_abs_value
                    min_threshold = dynamic_range * 1e-4
                    threshold = max(relative_threshold, min_threshold)
                    
                    mask = delta.abs() > threshold
                    
                    if len(delta.shape) > 1:
                        flat_delta = delta.view(-1)
                        flat_mask = mask.view(-1)
                        indices = flat_mask.nonzero().squeeze()
                        values = flat_delta[indices]
                    else:
                        indices = mask.nonzero().squeeze()
                        values = delta[indices]
                    
                    if indices.numel() > 0:
                        # Calculate sizes
                        param_size = param.numel() * 4  # dense size in bytes
                        max_index = param.numel() - 1
                        index_bytes = 1 if max_index <= 255 else (2 if max_index <= 65535 else 4)
                        sparse_size = (indices.numel() * index_bytes + values.numel() * 4 + 3)  # sparse size with overhead
                        
                        total_dense_size += param_size
                        
                        if sparse_size < param_size:
                            # Store as sparse update
                            self.weight_deltas[name] = (indices, values)
                            total_sparse_size += sparse_size
                        else:
                            # Store as dense update
                            self.weight_deltas[name] = (None, delta.cpu())
                            total_sparse_size += param_size
                    else:
                        # No significant updates, store as dense update with zeros
                        self.weight_deltas[name] = (None, torch.zeros_like(delta).cpu())
                        total_sparse_size += param.numel() * 4
        
        # Update communication metrics
        self.communication_metrics['model_size_bytes'] = total_dense_size
        self.communication_metrics['download_bytes'] = total_dense_size
        self.communication_metrics['upload_bytes'] = total_sparse_size
        self.communication_metrics['compression_ratio'] = total_dense_size / total_sparse_size if total_sparse_size > 0 else 1.0
        
        # Calculate memory saved
        self.last_mem_saved = max(0, total_dense_size - total_sparse_size)
        
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