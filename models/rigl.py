import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math


class RigLMask:
    """
    Manages binary masks for RigL (Rigged Lottery) sparse training.
    
    This class handles the creation, updating, and application of binary masks
    for implementing the RigL algorithm, which dynamically changes the sparse
    connectivity pattern during training.
    """
    
    def __init__(self, tensor: torch.Tensor, sparsity: float):
        """
        Initialize a binary mask for the given tensor with specified sparsity.
        
        Args:
            tensor: The parameter tensor to create a mask for
            sparsity: The fraction of elements to be masked (0.0 to 1.0)
        """
        self.shape = tensor.shape
        self.device = tensor.device
        self.sparsity = sparsity
        
        # Initialize the binary mask with random pruning
        self.mask = self._initialize_mask(tensor, sparsity)
        
        # Track statistics
        self.total_params = tensor.numel()
        self.nonzero_params = int(self.total_params * (1 - sparsity))
        
    def _initialize_mask(self, tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
        """
        Create an initial binary mask by randomly pruning weights.
        
        Args:
            tensor: The parameter tensor to create a mask for
            sparsity: The fraction of elements to be masked (0.0 to 1.0)
            
        Returns:
            A binary mask tensor with the same shape as the input tensor
        """
        # Ensure valid sparsity
        sparsity = max(0.0, min(1.0, sparsity))
        
        # Create a random mask
        mask = torch.rand_like(tensor, device=self.device)
        
        # Calculate threshold for keeping top (1-sparsity) fraction of elements
        if sparsity < 1.0:
            threshold = torch.quantile(mask.flatten(), sparsity)
            mask = (mask > threshold).float()
        else:
            mask = torch.zeros_like(tensor, device=self.device)
            
        return mask
    
    def update(self, tensor: torch.Tensor, gradient: torch.Tensor, new_sparsity: Optional[float] = None) -> torch.Tensor:
        """
        Update the mask using the RigL algorithm.
        
        This implements the core RigL logic:
        1. Remove connections with the smallest magnitude weights
        2. Add new connections where gradients have the largest magnitude
        
        Args:
            tensor: The parameter tensor
            gradient: The gradient tensor
            new_sparsity: Optional new sparsity level to target (if None, use current sparsity)
            
        Returns:
            The updated binary mask
        """
        # Update sparsity if specified
        if new_sparsity is not None:
            self.sparsity = max(0.0, min(1.0, new_sparsity))
        
        # Calculate number of connections to prune/regrow
        n_params = tensor.numel()
        n_zeros = int(n_params * self.sparsity)
        n_ones = n_params - n_zeros
        
        # Flatten tensors for easier manipulation
        flat_tensor = tensor.flatten()
        flat_grad = gradient.flatten()
        flat_mask = self.mask.flatten()
        
        # Find connections to remove (smallest magnitude weights)
        # Only consider weights that are currently active (mask == 1)
        active_weights = torch.where(flat_mask > 0.5, torch.abs(flat_tensor), torch.ones_like(flat_tensor) * float('inf'))
        _, prune_indices = torch.topk(active_weights, n_ones, largest=False)
        
        # Find connections to add (largest magnitude gradients)
        # Only consider weights that are currently inactive (mask == 0)
        inactive_grads = torch.where(flat_mask < 0.5, torch.abs(flat_grad), torch.zeros_like(flat_grad))
        _, grow_indices = torch.topk(inactive_grads, n_zeros, largest=True)
        
        # Create new mask
        new_mask = torch.zeros_like(flat_mask)
        new_mask[grow_indices] = 1.0
        
        # Keep the remaining active connections
        remaining_indices = torch.ones_like(flat_mask).bool()
        remaining_indices[prune_indices] = False
        remaining_indices[grow_indices] = False
        
        # Combine masks
        new_mask = new_mask + (flat_mask * remaining_indices.float())
        
        # Reshape mask back to original tensor shape
        self.mask = new_mask.reshape(self.shape)
        
        # Update statistics
        self.nonzero_params = int(self.total_params * (1 - self.sparsity))
        
        return self.mask
    
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the binary mask to a tensor.
        
        Args:
            tensor: The tensor to apply the mask to
            
        Returns:
            The masked tensor
        """
        return tensor * self.mask


class RigLOptimizer:
    """
    Implements the RigL (Rigged Lottery) optimization algorithm for sparse training.
    
    This optimizer wraps an existing optimizer and applies the RigL algorithm to
    dynamically change the sparse connectivity pattern during training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        initial_sparsity: float = 0.5,
        target_sparsity: float = 0.9,
        update_interval: int = 100,
        final_update_epoch: int = 100,
        T_end: Optional[int] = None,
        dense_layers: List[str] = None
    ):
        """
        Initialize the RigL optimizer.
        
        Args:
            optimizer: Base optimizer (e.g., SGD, Adam)
            model: The neural network model
            initial_sparsity: Initial sparsity level (0.0 to 1.0)
            target_sparsity: Final target sparsity level (0.0 to 1.0)
            update_interval: Number of steps between mask updates
            final_update_epoch: Epoch after which to stop updating masks
            T_end: Total number of training steps (if None, calculated from final_update_epoch)
            dense_layers: List of layer names to keep dense (no pruning)
        """
        self.optimizer = optimizer
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.update_interval = update_interval
        self.final_update_epoch = final_update_epoch
        self.T_end = T_end
        self.dense_layers = dense_layers or []
        
        # Initialize step counter
        self.step_count = 0
        self.epoch = 0
        
        # Initialize masks for each parameter
        self.masks = {}
        self._initialize_masks()
        
        # Apply initial masks
        self._apply_masks()
        
        # Track statistics
        self.stats = {
            "total_params": 0,
            "nonzero_params": 0,
            "sparsity": 0.0,
            "layer_sparsity": {},
            "mask_updates": 0
        }
        self._update_stats()
    
    def _initialize_masks(self):
        """Initialize binary masks for all parameters in the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) > 1 and name not in self.dense_layers:
                self.masks[name] = RigLMask(param.data, self.initial_sparsity)
                # Apply initial mask to parameter
                param.data *= self.masks[name].mask
    
    def _apply_masks(self):
        """Apply binary masks to all parameters in the model."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].mask
    
    def _update_masks(self):
        """Update binary masks according to the RigL schedule."""
        # Only update if we haven't reached the final update epoch
        if self.epoch >= self.final_update_epoch:
            return
        
        # Calculate current sparsity level using cosine schedule
        if self.T_end is None:
            # Estimate T_end based on final_update_epoch
            self.T_end = self.final_update_epoch * self.update_interval
        
        # Calculate current sparsity using cosine annealing
        progress = min(1.0, self.step_count / self.T_end)
        current_sparsity = self.initial_sparsity + 0.5 * (self.target_sparsity - self.initial_sparsity) * (1 + math.cos(math.pi * (1 - progress)))
        
        # Update masks for each parameter
        for name, param in self.model.named_parameters():
            if name in self.masks and param.grad is not None:
                self.masks[name].update(param.data, param.grad.data, current_sparsity)
        
        # Apply updated masks
        self._apply_masks()
        
        # Update statistics
        self._update_stats()
    
    def _update_stats(self):
        """Update sparsity statistics."""
        total_params = 0
        nonzero_params = 0
        layer_sparsity = {}
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                total_params += mask.total_params
                nonzero_params += mask.nonzero_params
                layer_sparsity[name] = 1.0 - (mask.nonzero_params / mask.total_params)
        
        self.stats["total_params"] = total_params
        self.stats["nonzero_params"] = nonzero_params
        self.stats["sparsity"] = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
        self.stats["layer_sparsity"] = layer_sparsity
    
    def zero_grad(self):
        """Zero out the gradients."""
        self.optimizer.zero_grad()
    
    def step(self):
        """Perform a single optimization step."""
        # Apply masks to gradients
        for name, param in self.model.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad.data *= self.masks[name].mask
        
        # Perform optimizer step
        self.optimizer.step()
        
        # Apply masks to parameters
        self._apply_masks()
        
        # Update masks if needed
        self.step_count += 1
        if self.step_count % self.update_interval == 0:
            self._update_masks()
            self.stats["mask_updates"] += 1
    
    def set_epoch(self, epoch):
        """Update the current epoch."""
        self.epoch = epoch
    
    def get_sparsity(self) -> float:
        """Get the current overall sparsity level."""
        return self.stats["sparsity"]
    
    def get_layer_sparsity(self) -> Dict[str, float]:
        """Get the current sparsity level for each layer."""
        return self.stats["layer_sparsity"]


class RigLTrainingHooks:
    """
    Provides hooks for integrating RigL with the training loop.
    
    This class offers callback functions that can be used to integrate
    RigL with existing training loops without modifying them extensively.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        initial_sparsity: float = 0.5,
        target_sparsity: float = 0.9,
        update_interval: int = 100,
        final_update_epoch: int = 100,
        dense_layers: List[str] = None
    ):
        """
        Initialize RigL training hooks.
        
        Args:
            model: The neural network model
            optimizer: Base optimizer (e.g., SGD, Adam)
            initial_sparsity: Initial sparsity level (0.0 to 1.0)
            target_sparsity: Final target sparsity level (0.0 to 1.0)
            update_interval: Number of steps between mask updates
            final_update_epoch: Epoch after which to stop updating masks
            dense_layers: List of layer names to keep dense (no pruning)
        """
        self.rigl_optimizer = RigLOptimizer(
            optimizer=optimizer,
            model=model,
            initial_sparsity=initial_sparsity,
            target_sparsity=target_sparsity,
            update_interval=update_interval,
            final_update_epoch=final_update_epoch,
            dense_layers=dense_layers
        )
        self.model = model
        
    def on_epoch_begin(self, epoch: int):
        """
        Called at the beginning of each epoch.
        
        Args:
            epoch: The current epoch number
        """
        self.rigl_optimizer.set_epoch(epoch)
    
    def on_batch_begin(self):
        """Called before each batch."""
        pass
    
    def on_batch_end(self):
        """Called after each batch."""
        pass
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        pass
    
    def get_sparsity_stats(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Get current sparsity statistics.
        
        Returns:
            A dictionary containing sparsity statistics
        """
        return {
            "sparsity": self.rigl_optimizer.get_sparsity(),
            "layer_sparsity": self.rigl_optimizer.get_layer_sparsity(),
            "mask_updates": self.rigl_optimizer.stats["mask_updates"],
            "total_params": self.rigl_optimizer.stats["total_params"],
            "nonzero_params": self.rigl_optimizer.stats["nonzero_params"]
        }
