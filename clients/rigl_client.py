import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import flwr as fl
from models.config import get_tinyprop_config
from utils.adaptive_sparsification import AdaptiveSparsifier
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from utils.training_helpers import compute_adaptive_ratio, compute_sparsity_and_flops
from typing import Dict, Tuple, List, Optional
import os
import psutil
import pandas as pd
from torch.utils.data import DataLoader
from utils.flops_calculator import compute_model_flops
from utils.memory_calculator import estimate_model_memory
from models.rigl import RigLOptimizer
from clients.federated_client import FederatedClient


class FederatedRigLClient(FederatedClient):
    """
    Federated client implementation that uses RigL (Rigged Lottery) for sparse training.
    
    This client extends the base FederatedClient and integrates RigL for dynamic
    sparse training with pruning and regrowth of connections during training.
    """
    
    def __init__(
        self, 
        client_id: int, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        cfg: dict, 
        device: str = None, 
        dataset_name: str = None,
        rigl_initial_sparsity: float = 0.5,
        rigl_target_sparsity: float = 0.9,
        rigl_update_interval: int = 100,
        rigl_final_update_epoch: int = 100
    ):
        """
        Initialize a federated client with RigL sparse training.
        
        Args:
            client_id: Unique identifier for this client
            model: Neural network model
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            cfg: Configuration dictionary
            device: Device to use for training ('cuda' or 'cpu')
            dataset_name: Name of the dataset
            rigl_initial_sparsity: Initial sparsity level for RigL
            rigl_target_sparsity: Target sparsity level for RigL
            rigl_update_interval: Number of steps between mask updates
            rigl_final_update_epoch: Epoch after which to stop updating masks
        """
        # Initialize the base FederatedClient
        super().__init__(client_id, model, train_loader, test_loader, cfg, device, dataset_name)
        
        # Store RigL parameters
        self.rigl_initial_sparsity = rigl_initial_sparsity
        self.rigl_target_sparsity = rigl_target_sparsity
        self.rigl_update_interval = rigl_update_interval
        self.rigl_final_update_epoch = rigl_final_update_epoch
        
        # Flag to indicate this is a RigL client
        self.is_rigl = True
        self.is_dense_baseline = False
        self.is_fedprune = False
        
        # Create RigL optimizer
        self.base_optimizer = self.optimizer  # Store the original optimizer
        self.rigl_optimizer = RigLOptimizer(
            optimizer=self.optimizer,
            model=self.model,
            initial_sparsity=self.rigl_initial_sparsity,
            target_sparsity=self.rigl_target_sparsity,
            update_interval=self.rigl_update_interval,
            final_update_epoch=self.rigl_final_update_epoch
        )
        
        # Replace the optimizer with RigL optimizer
        self.optimizer = self.rigl_optimizer
        
        # Track RigL-specific metrics
        self.rigl_metrics = {
            "sparsity": [],
            "mask_updates": 0,
            "current_sparsity": self.rigl_initial_sparsity
        }
        
        # Initialize layer sparsity for metrics
        self.layer_sparsity = {}
        
    def train(
        self, 
        num_epochs: int = 1, 
        batch_size: int = 32, 
        global_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[float, float]:
        """
        Train the model on the local dataset.
        
        Args:
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            global_params: Global model parameters for FedProx regularization
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        device = next(self.model.parameters()).device
        
        # Reset metrics for this round
        losses = []
        accuracies = []
        grad_norms = []
        self.last_flops = 0  # Reset FLOPs counter for this training round
        
        # Reset counters
        self.num_skipped_batches = 0
        self.total_batches = 0
        
        # FedProx regularization
        mu = self.cfg.get("fedprox_mu", 0.01) if self.cfg.get("use_fedprox", False) else 0.0
        
        # Track initial parameters for delta calculation
        if self.global_weights is None:
            self.global_weights = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        else:
            for name, param in self.model.named_parameters():
                if name in self.global_weights:
                    self.global_weights[name] = param.clone().detach()
        
        # Training loop
        for epoch in range(num_epochs):
            # Set current epoch for RigL optimizer
            if hasattr(self.optimizer, 'set_epoch'):
                self.optimizer.set_epoch(epoch)
            
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # FedProx regularization if enabled
                if mu > 0 and global_params is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            proximal_term += torch.sum((param - global_params[name].to(device)) ** 2)
                    loss += (mu / 2) * proximal_term
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norm for metrics
                grad_norm = self.compute_grad_norm()
                grad_norms.append(grad_norm)
                
                # Update weights using RigL optimizer
                self.optimizer.step()
                
                # Update layer_sparsity from RigL optimizer's masks if available
                if hasattr(self.optimizer, 'masks'):
                    # Create a mapping between parameter names and module names
                    param_to_module = {}
                    for name, module in self.model.named_modules():
                        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                            for param_name, _ in module.named_parameters():
                                full_param_name = f"{name}.{param_name}"
                                param_to_module[full_param_name] = name
                    
                    # Update layer_sparsity using the mapping
                    for param_name, mask_obj in self.optimizer.masks.items():
                        if mask_obj is not None:
                            # Find the corresponding module name
                            for full_param_name, module_name in param_to_module.items():
                                if param_name in full_param_name or full_param_name.endswith(param_name):
                                    # Access the actual mask tensor via the .mask attribute
                                    self.layer_sparsity[module_name] = 1.0 - (mask_obj.mask.mean().item())
                                    break
                
                # Calculate FLOPs for this batch using initial sparsity (constant FLOPs as per RigL paper)
                # RigL maintains constant FLOPs throughout training regardless of changing sparsity
                from utils.flops_calculator import compute_model_flops
                
                # Create a constant sparsity dictionary based on initial sparsity
                constant_layer_sparsity = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                        constant_layer_sparsity[name] = self.rigl_initial_sparsity
                
                # Use constant sparsity for FLOPs calculation to maintain constant FLOPs
                batch_flops, _ = compute_model_flops(self.model, data.shape, layer_sparsity_dict=constant_layer_sparsity)
                self.last_flops += batch_flops  # Accumulate FLOPs
                
                # Update metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Increment batch counter
                self.total_batches += 1
                
            # Calculate epoch metrics
            epoch_loss /= len(self.train_loader)
            epoch_accuracy = 100.0 * correct / total
            
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
            
            # Update learning rate scheduler if available
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Calculate average metrics
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        
        # Store metrics
        self.last_avg_grad_norm = avg_grad_norm
        
        # Calculate weight deltas for sparse aggregation
        self.weight_deltas = self._calculate_weight_deltas()
        
        # Update RigL metrics
        if hasattr(self.optimizer, 'get_sparsity'):
            self.rigl_metrics["current_sparsity"] = self.optimizer.get_sparsity()
            self.rigl_metrics["sparsity"].append(self.rigl_metrics["current_sparsity"])
            self.layer_sparsity = self.optimizer.get_layer_sparsity()
            self.rigl_metrics["mask_updates"] = self.optimizer.stats["mask_updates"]
        
        # Update last sparsity for metrics
        self.last_sparsity = self.rigl_metrics["current_sparsity"]
        
        # Compute comprehensive metrics
        self.compute_metrics()
        
        return avg_loss, avg_accuracy
    
    def _calculate_weight_deltas(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calculate sparse weight deltas for efficient communication.
        
        Returns:
            Dictionary mapping parameter names to tuples of (indices, values)
        """
        deltas = {}
        
        for name, param in self.model.named_parameters():
            if name in self.global_weights:
                # Calculate delta from global weights
                delta = param.data - self.global_weights[name].to(param.device)
                
                # Apply sparsification based on RigL mask if available
                if hasattr(self.optimizer, 'masks') and name in self.optimizer.masks:
                    # Use RigL mask for sparsification
                    mask = self.optimizer.masks[name].mask
                    sparse_delta = delta * mask
                else:
                    # Fallback to magnitude-based sparsification
                    k = max(1, int(delta.numel() * (1 - self.rigl_metrics["current_sparsity"])))
                    values, indices = torch.topk(torch.abs(delta.view(-1)), k)
                    threshold = values[-1]
                    mask = (torch.abs(delta) >= threshold).float()
                    sparse_delta = delta * mask
                
                # Create sparse representation
                nonzero_indices = torch.nonzero(sparse_delta.view(-1), as_tuple=True)[0]
                nonzero_values = sparse_delta.view(-1)[nonzero_indices]
                
                if len(nonzero_indices) > 0:
                    deltas[name] = (nonzero_indices, nonzero_values)
                else:
                    # If no nonzero elements, use a special marker
                    deltas[name] = (None, delta)
        
        return deltas
    
    def local_evaluate(self, test_loader: DataLoader) -> float:
        """
        Evaluate the model on the local test dataset.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Accuracy as a percentage
        """
        self.model.eval()
        correct = 0
        total = 0
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total if total > 0 else 0.0
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        
        # Update layer_sparsity from RigL optimizer's masks if available
        if hasattr(self.optimizer, 'masks'):
            # Create a mapping between parameter names and module names
            param_to_module = {}
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    for param_name, _ in module.named_parameters():
                        full_param_name = f"{name}.{param_name}"
                        param_to_module[full_param_name] = name
            
            # Update layer_sparsity using the mapping
            for param_name, mask_obj in self.optimizer.masks.items():
                if mask_obj is not None:
                    # Find the corresponding module name
                    for full_param_name, module_name in param_to_module.items():
                        if param_name in full_param_name or full_param_name.endswith(param_name):
                            # Access the actual mask tensor via the .mask attribute
                            self.layer_sparsity[module_name] = 1.0 - (mask_obj.mask.mean().item())
                            break
        
        # Calculate FLOPs for this batch using initial sparsity (constant FLOPs as per RigL paper)
        # RigL maintains constant FLOPs throughout training regardless of changing sparsity
        from utils.flops_calculator import compute_model_flops
        
        # Create a constant sparsity dictionary based on initial sparsity
        constant_layer_sparsity = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                constant_layer_sparsity[name] = self.rigl_initial_sparsity
        
        # Use constant sparsity for FLOPs calculation to maintain constant FLOPs
        batch_flops, _ = compute_model_flops(self.model, data.shape, layer_sparsity_dict=constant_layer_sparsity)
        self.last_flops += batch_flops  # Accumulate FLOPs
        
        Returns:
            Dictionary of metrics
        """
        metrics = super().get_metrics()
        
        # Add RigL-specific metrics
        metrics.update({
            'rigl_sparsity': self.rigl_metrics["current_sparsity"],
            'rigl_mask_updates': self.rigl_metrics["mask_updates"],
            'sparsity': self.last_sparsity,
            'download_bytes': self.communication_metrics['download_bytes'],
            'upload_bytes': self.communication_metrics['upload_bytes'],
            'model_size_bytes': self.communication_metrics['model_size_bytes'],
            'compression_ratio': self.communication_metrics['compression_ratio'],
            'flops': self.last_flops
        })
        
        return metrics
