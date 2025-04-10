import torch
import numpy as np
from typing import Dict, Optional, Tuple

class AdaptiveSparsifier:
    def __init__(
        self,
        initial_sparsity: float = 0.3,
        target_sparsity: float = 0.9,
        warmup_rounds: int = 5,
        total_rounds: int = 100,
        min_accuracy: float = 0.85,
        energy_budget: Optional[float] = None
    ):
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.warmup_rounds = warmup_rounds
        self.total_rounds = total_rounds
        self.min_accuracy = min_accuracy
        self.energy_budget = energy_budget
        
        # History tracking
        self.accuracy_history = []
        self.sparsity_history = []
        self.energy_history = []
        
        # Layer-specific tracking
        self.layer_sensitivity: Dict[str, float] = {}
        self.layer_importance: Dict[str, float] = {}
    
    def compute_round_sparsity(
        self,
        round_num: int,
        current_accuracy: float,
        current_energy: Optional[float] = None
    ) -> float:
        """Compute adaptive sparsity for the current round."""
        # During warmup, use initial sparsity
        if round_num < self.warmup_rounds:
            return self.initial_sparsity
        
        # Update history
        self.accuracy_history.append(current_accuracy)
        if current_energy is not None:
            self.energy_history.append(current_energy)
        
        # Compute progress factor (0 to 1)
        progress = (round_num - self.warmup_rounds) / (self.total_rounds - self.warmup_rounds)
        
        # Base sparsity from linear schedule
        base_sparsity = self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * progress
        
        # Adjust based on accuracy
        accuracy_factor = self._compute_accuracy_factor(current_accuracy)
        
        # Adjust based on energy budget if provided
        energy_factor = 1.0
        if self.energy_budget and current_energy:
            energy_factor = self._compute_energy_factor(current_energy)
        
        # Combine factors
        adjusted_sparsity = base_sparsity * accuracy_factor * energy_factor
        
        # Ensure sparsity stays within bounds
        return max(self.initial_sparsity, min(self.target_sparsity, adjusted_sparsity))
    
    def compute_layer_sparsity(
        self,
        model: torch.nn.Module,
        global_sparsity: float
    ) -> Dict[str, float]:
        """Compute layer-wise sparsity based on layer sensitivity and importance."""
        # Initialize layer sensitivity if not computed
        if not self.layer_sensitivity:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Initialize with uniform sensitivity
                    self.layer_sensitivity[name] = 1.0
        
        # Update layer importance
        self._update_layer_importance(model)
        
        # Compute layer-specific sparsity
        layer_sparsity = {}
        total_params = sum(p.numel() for p in model.parameters())
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                sensitivity = self.layer_sensitivity.get(name, 1.0)
                importance = self.layer_importance.get(name, 1.0)
                
                # More important or sensitive layers get lower sparsity
                layer_factor = (1 - sensitivity) * (1 - importance)
                layer_sparsity[name] = global_sparsity * (1 + layer_factor)
                
                # Ensure sparsity stays within bounds
                layer_sparsity[name] = max(0.0, min(0.95, layer_sparsity[name]))
        
        return layer_sparsity
    
    def _compute_accuracy_factor(self, current_accuracy: float) -> float:
        """Compute factor to adjust sparsity based on accuracy."""
        if current_accuracy < self.min_accuracy:
            # Reduce sparsity if accuracy is too low
            return 0.8
        elif len(self.accuracy_history) >= 2:
            # Check if accuracy is improving
            accuracy_change = current_accuracy - self.accuracy_history[-2]
            if accuracy_change > 0.01:
                # Accuracy improving, can increase sparsity
                return 1.1
            elif accuracy_change < -0.01:
                # Accuracy decreasing, reduce sparsity
                return 0.9
        return 1.0
    
    def _compute_energy_factor(self, current_energy: float) -> float:
        """Compute factor to adjust sparsity based on energy consumption."""
        if not self.energy_budget:
            return 1.0
            
        energy_per_round = self.energy_budget / self.total_rounds
        if current_energy > energy_per_round:
            # Over budget, increase sparsity
            return 1.2
        return 1.0
    
    def _compute_layer_sensitivity(self, model: torch.nn.Module) -> None:
        """Compute layer sensitivity based on gradient magnitude."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_magnitude = param.grad.abs().mean().item()
                self.layer_sensitivity[name] = grad_magnitude
        
        # Normalize sensitivity values
        max_sensitivity = max(self.layer_sensitivity.values())
        for name in self.layer_sensitivity:
            self.layer_sensitivity[name] /= max_sensitivity
    
    def _update_layer_importance(self, model: torch.nn.Module) -> None:
        """Update layer importance based on parameter magnitude and position."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Compute importance based on parameter magnitude
                param_importance = param.abs().mean().item()
                
                # Layer position factor (earlier layers are more important)
                layer_depth = len(name.split('.'))
                position_factor = 1.0 / layer_depth
                
                self.layer_importance[name] = param_importance * position_factor
        
        # Normalize importance values
        max_importance = max(self.layer_importance.values())
        for name in self.layer_importance:
            self.layer_importance[name] /= max_importance
    
    def get_sparsification_stats(self) -> Dict:
        """Get statistics about the sparsification process."""
        return {
            'current_sparsity': self.sparsity_history[-1] if self.sparsity_history else self.initial_sparsity,
            'accuracy_trend': self.accuracy_history[-5:] if len(self.accuracy_history) >= 5 else self.accuracy_history,
            'layer_sensitivity': self.layer_sensitivity,
            'layer_importance': self.layer_importance,
            'energy_efficiency': sum(self.energy_history) / len(self.energy_history) if self.energy_history else None
        } 