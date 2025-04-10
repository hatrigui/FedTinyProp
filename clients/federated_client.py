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

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader=None, device="cpu", dataset_name="mnist"):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = get_tinyprop_config(dataset_name)
        # Ensure the FLOPs per batch is defined (adjust the default if needed)
        self.cfg.setdefault("full_flops_per_batch", 1e6)
        
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.cfg["optimizer"]["lr"],
            momentum=self.cfg["optimizer"].get("momentum", 0.9),
            weight_decay=self.cfg["optimizer"].get("weight_decay", 0.0)
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Adaptive sparsity configuration
        self.S_min = self.cfg.get("S_min", 0.05)  # Minimum sparsity
        self.S_max = self.cfg.get("S_max", 0.5)   # Maximum sparsity
        self.zeta = self.cfg.get("zeta", 0.25)    # Scaling factor for phi
        self.phi_skip_threshold = self.cfg.get("phi_skip_threshold", 0.2)  # New threshold for skipping updates
        self.grad_norm_threshold = self.cfg.get("grad_norm_threshold", 1e-3)
        self.skip_threshold = self.cfg.get("skip_threshold", 2.5)  # Set appropriate skip threshold
        
        self.sparsifier = AdaptiveSparsifier(
            initial_sparsity=self.cfg.get("initial_sparsity", 0.3),
            target_sparsity=self.cfg.get("target_sparsity", 0.9),
            total_rounds=self.cfg.get("total_rounds", 100),
            energy_budget=self.cfg.get("energy_budget", None)
        )
        
        # Initialize metrics tracking
        self.weight_deltas = {}
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
        self.initial_grad_norms = []
        self.INITIAL_GRAD_NORM_BATCHES = 5
        self.adaptive_sparsity = True
        self.phi_min = self.cfg.get("phi_min", 0.1)
        
        # Add EMA smoothing parameters
        self.phi_ema_alpha = 0.9  # EMA coefficient for phi smoothing
        self.target_sparsity_adjustment_rate = 0.1  # Rate to adjust towards target sparsity
        
        # Enhanced metrics tracking
        self.metrics = {
            "nonzero_gradients": [],
            "batch_flops": [],
            "peak_memory": 0,
            "communication_cost": 0,
            "effective_sparsity": [],
            "phi_values": [],
            "skipped_updates": 0
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
        num_epochs = config.get("num_epochs", 1) if config else 1
        
        # Train the model
        loss, accuracy = self.train(num_epochs=num_epochs, batch_size=batch_size)
        
        # Get updated parameters and metrics
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
            return float(acc), len(self.test_loader.dataset)  # Return only accuracy and num_examples
        return 0.0, 0  # Return only accuracy and num_examples

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
            # If the very first gradient norm is near zero, set it to epsilon
            if abs(current_grad_norm) < epsilon:
                self.initial_grad_norm = epsilon
            else:
                self.initial_grad_norm = current_grad_norm
            self.smoothed_phi = 1.0
            return 1.0
        
        # Compute raw phi
        raw_phi = compute_adaptive_ratio(current_grad_norm, self.initial_grad_norm, self.phi_min)
        
        # Apply EMA smoothing
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
        """Apply gradient sparsification with random masking."""
        total_nonzero = 0
        total_elements = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Compute layer sparsity based on phi
                layer_sparsity = max(self.S_min, min(self.S_max, self.zeta * phi))
                
                # Use random masking: for each element, drop it with probability equal to sparsity
                mask = torch.rand_like(param) > layer_sparsity
                param.grad.data *= mask.float()
                
                # Track metrics
                nonzero_count = mask.sum().item()
                total_nonzero += nonzero_count
                total_elements += mask.numel()
                
                # Store layer sparsity for metrics
                self.layer_sparsity[name] = 1.0 - (nonzero_count / mask.numel())
        
        # Update metrics
        if total_elements > 0:
            effective_sparsity = 1.0 - (total_nonzero / total_elements)
            self.metrics["nonzero_gradients"].append(total_nonzero)
            self.metrics["effective_sparsity"].append(effective_sparsity)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive training metrics."""
        metrics = {
            "flops": 0.0,
            "memory": 0.0,
            "communication": 0.0,
            "sparsity": 0.0,
            "avg_nonzero_gradients": 0.0,
            "avg_effective_sparsity": 0.0,
            "peak_memory_mb": 0.0,
            "communication_bytes": 0
        }
        
        # Compute sparsity and FLOPs
        sparsity, flops = compute_sparsity_and_flops(self.model, self.cfg["full_flops_per_batch"])
        metrics["flops"] = flops
        metrics["sparsity"] = sparsity
        
        # Add enhanced metrics if available
        if self.metrics["nonzero_gradients"]:
            metrics["avg_nonzero_gradients"] = np.mean(self.metrics["nonzero_gradients"])
        if self.metrics["effective_sparsity"]:
            metrics["avg_effective_sparsity"] = np.mean(self.metrics["effective_sparsity"])
        
        # Track memory usage (GPU or CPU)
        if torch.cuda.is_available():
            metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process(os.getpid())
            metrics["peak_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        
        # Compute communication cost based on sparse weight deltas
        total_bytes = 0
        for name, (indices, values) in self.weight_deltas.items():
            if isinstance(indices, torch.Tensor) and isinstance(values, torch.Tensor):
                # Each index is 4 bytes (int32) and each value is 4 bytes (float32)
                total_bytes += (indices.numel() + values.numel()) * 4
        
        metrics["communication_bytes"] = total_bytes
        
        # Update compression ratio
        if total_bytes > 0:
            # Calculate original size (all parameters)
            original_size = sum(p.numel() * 4 for p in self.model.parameters() if p.requires_grad)
            self.compression_ratio = total_bytes / original_size
        else:
            self.compression_ratio = 1.0  # No compression if no updates
        
        return metrics

    def train(self, num_epochs: int, batch_size: int) -> Tuple[float, float]:
        """Train the model on the local dataset."""
        print(f"\n[Client Debug] Starting training for {num_epochs} epochs with batch size {batch_size}")
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Store initial weights for computing deltas
        initial_weights = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        
        # Initialize metrics
        epoch_grad_norms = []
        epoch_skipped_batches = 0
        total_flops = 0
        total_memory = 0
        total_communication = 0
        epoch_effective_sparsities = []
        
        # Get current round from the model's state if available
        current_round = getattr(self.model, 'current_round', 0)
        total_rounds = self.cfg.get("total_rounds", 100)
        
        # Compute progressive sparsity factor based on current round
        progress = current_round / total_rounds
        progressive_factor = 1.0 + (self.S_max - self.S_min) * progress
        
        for epoch in range(num_epochs):
            batch_grad_norms = []
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.model.zero_grad()
                loss.backward()
                
                # Compute gradient norm using existing method
                grad_norm = self.compute_grad_norm()
                if grad_norm > 0:
                    batch_grad_norms.append(grad_norm)
                    
                    # Update adaptive sparsity and compute phi
                    if self.adaptive_sparsity:
                        phi = self.update_adaptive_sparsity(grad_norm)
                        
                        # Skip batch with probability (1 - phi)
                        if torch.rand(1).item() > phi:
                            epoch_skipped_batches += 1
                            self.metrics["skipped_updates"] += 1
                            continue
                        
                        # Apply gradient sparsification with random masking
                        self.apply_gradient_sparsification(phi)
                        
                        # Compute progressive sparsity that increases over time
                        base_sparsity = self.S_min + (self.S_max - self.S_min) * (1 - phi)
                        local_sparsity = min(self.S_max, base_sparsity * progressive_factor)
                        epoch_effective_sparsities.append(local_sparsity)
                        
                        # Compute FLOPs for this batch
                        batch_flops = self.cfg["full_flops_per_batch"] * (1 - local_sparsity)
                        total_flops += batch_flops
                        self.metrics["batch_flops"].append(batch_flops)
                
                # Update weights
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Track memory usage (GPU or CPU)
                if torch.cuda.is_available():
                    total_memory = max(total_memory, torch.cuda.max_memory_allocated())
                else:
                    process = psutil.Process(os.getpid())
                    total_memory = max(total_memory, process.memory_info().rss)
            
            # Store average gradient norm for this epoch
            if batch_grad_norms:
                epoch_grad_norms.append(sum(batch_grad_norms) / len(batch_grad_norms))
            
            # Update client's summary metrics
            self.last_flops = total_flops / len(self.train_loader)
            self.last_mem = total_memory / (1024 * 1024)  # Convert to MB
            self.num_skipped_batches = epoch_skipped_batches  # Track actual skipped batches
            self.last_avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
            self.last_sparsity = np.mean(epoch_effective_sparsities) if epoch_effective_sparsities else self.cfg.get("initial_sparsity", 0.3)
            
            # Print epoch stats
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  - Average Loss: {total_loss / (batch_idx + 1):.4f}")
            print(f"  - Accuracy: {100. * correct / total:.2f}%")
            if self.adaptive_sparsity:
                print(f"  - Smoothed Phi: {self.last_phi:.4f}")
                print(f"  - Skipped Batches: {epoch_skipped_batches}")
                print(f"  - Current Sparsity: {self.last_sparsity:.4f}")
            if epoch_grad_norms:
                print(f"  - Average Gradient Norm: {epoch_grad_norms[-1]:.4f}")
            print(f"  - Peak Memory: {self.last_mem:.1f}MB")
        
        # Compute weight deltas and store them sparsely
        total_sparse_bytes = 0
        total_original_bytes = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta = param.data - initial_weights[name]
                # Flatten the absolute delta to compute threshold and mask consistently
                flat_delta = delta.abs().flatten()
                # Get indices of values above threshold using percentile
                k = max(1, int(0.1 * flat_delta.numel()))  # Keep top 10%
                threshold = torch.kthvalue(flat_delta, flat_delta.numel() - k + 1)[0]
                # Compute mask on flattened tensor to ensure 1D indices
                mask = flat_delta >= threshold
                
                # Calculate original size (4 bytes per element)
                total_original_bytes += delta.numel() * 4
                
                if mask.sum() > 0:
                    # Get linear indices of nonzero elements (will be 1D)
                    indices = mask.nonzero().squeeze(1)
                    # Store values at those indices
                    values = delta.flatten()[indices]
                    # Store as 2-tuple (indices, values)
                    self.weight_deltas[name] = (indices, values)
                    # Calculate sparse size (4 bytes per index + 4 bytes per value)
                    total_sparse_bytes += (indices.numel() + values.numel()) * 4
                else:
                    # Store empty tensors if no significant changes
                    self.weight_deltas[name] = (
                        torch.tensor([], dtype=torch.long), 
                        torch.tensor([], device=delta.device)
                    )
        
        # Update compression ratio and communication bytes based on actual sparse representation
        if total_original_bytes > 0:
            self.compression_ratio = total_sparse_bytes / total_original_bytes
            self.last_comm = total_sparse_bytes  # Update communication bytes
        else:
            self.compression_ratio = 1.0  # No compression if no updates
            self.last_comm = total_original_bytes  # Use original size if no updates
        
        # Return average loss and accuracy
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else float('inf')
        accuracy = 100. * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

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
