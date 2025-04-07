import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class EnergyMetrics:
    compute_energy: float
    memory_energy: float
    communication_energy: float
    
    @property
    def total_energy(self) -> float:
        return self.compute_energy + self.memory_energy + self.communication_energy

@dataclass
class HardwareConstraints:
    memory_limit: int  # in bytes
    compute_capability: float  # FLOPS
    energy_budget: float  # in Joules
    bandwidth: float  # in bytes/second

class TinyMLMetrics:
    def __init__(self, hardware_constraints: Optional[HardwareConstraints] = None):
        # Energy constants (in Joules)
        self.ENERGY_PER_FLOP = 1e-9
        self.ENERGY_PER_MEMORY_ACCESS = 1e-8
        self.ENERGY_PER_BYTE_TRANSMITTED = 1e-7
        
        # History tracking
        self.energy_history: List[EnergyMetrics] = []
        self.memory_profile: List[Dict] = []
        self.communication_stats: List[Dict] = []
        self.convergence_info: List[Dict] = []
        self.sparsity_stats: List[Dict] = []
        
        # Hardware constraints
        self.hardware = hardware_constraints
    
    def estimate_energy_consumption(self, flops: int, memory_access: int, communication_bytes: int) -> EnergyMetrics:
        compute_energy = flops * self.ENERGY_PER_FLOP
        memory_energy = memory_access * self.ENERGY_PER_MEMORY_ACCESS
        communication_energy = communication_bytes * self.ENERGY_PER_BYTE_TRANSMITTED
        
        return EnergyMetrics(
            compute_energy=compute_energy,
            memory_energy=memory_energy,
            communication_energy=communication_energy
        )
    
    def track_round_metrics(self, round_num: int, model: torch.nn.Module, 
                          sparsity: float, communication_bytes: int,
                          accuracy: float, loss: float) -> Dict:
        # Compute metrics
        flops = self._estimate_round_flops(model, sparsity)
        memory_access = self._estimate_memory_access(model)
        energy = self.estimate_energy_consumption(flops, memory_access, communication_bytes)
        
        # Track energy
        self.energy_history.append(energy)
        
        # Track memory usage
        memory_stats = {
            'peak_memory': memory_access * 4,  # assuming 4 bytes per parameter
            'working_set': self._estimate_working_set(model),
            'gradient_memory': self._estimate_gradient_memory(model, sparsity)
        }
        self.memory_profile.append(memory_stats)
        
        # Track communication
        comm_stats = {
            'bytes_sent': communication_bytes,
            'compression_ratio': self._compute_compression_ratio(model, sparsity),
            'bandwidth_utilization': self._estimate_bandwidth_utilization(communication_bytes)
        }
        self.communication_stats.append(comm_stats)
        
        # Track convergence
        convergence_stats = {
            'accuracy': accuracy,
            'loss': loss,
            'learning_rate': self._get_current_lr(model)
        }
        self.convergence_info.append(convergence_stats)
        
        # Track sparsity
        sparsity_stats = {
            'global_sparsity': sparsity,
            'layer_wise_sparsity': self._compute_layer_sparsity(model),
            'effective_compression': self._compute_effective_compression(sparsity)
        }
        self.sparsity_stats.append(sparsity_stats)
        
        return {
            'energy': energy,
            'memory': memory_stats,
            'communication': comm_stats,
            'convergence': convergence_stats,
            'sparsity': sparsity_stats
        }
    
    def _estimate_round_flops(self, model: torch.nn.Module, sparsity: float) -> int:
        total_params = sum(p.numel() for p in model.parameters())
        # Forward pass: full computation
        forward_flops = total_params * 2  # multiply-add per parameter
        # Backward pass: reduced by sparsity
        backward_flops = forward_flops * (1 - sparsity) * (1 - sparsity)  # quadratic reduction
        return forward_flops + backward_flops
    
    def _estimate_memory_access(self, model: torch.nn.Module) -> int:
        total_params = sum(p.numel() for p in model.parameters())
        # Each parameter is accessed multiple times
        return total_params * 3  # read param, read grad, write update
    
    def _estimate_working_set(self, model: torch.nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters())
    
    def _estimate_gradient_memory(self, model: torch.nn.Module, sparsity: float) -> int:
        total_params = sum(p.numel() for p in model.parameters())
        return int(total_params * (1 - sparsity) * 4)  # 4 bytes per gradient
    
    def _compute_compression_ratio(self, model: torch.nn.Module, sparsity: float) -> float:
        uncompressed_size = sum(p.numel() * p.element_size() for p in model.parameters())
        compressed_size = uncompressed_size * (1 - sparsity)
        return uncompressed_size / compressed_size if compressed_size > 0 else float('inf')
    
    def _estimate_bandwidth_utilization(self, bytes_sent: int) -> float:
        if self.hardware and self.hardware.bandwidth > 0:
            return bytes_sent / self.hardware.bandwidth
        return 0.0
    
    def _get_current_lr(self, model: torch.nn.Module) -> float:
        # Try to get learning rate from optimizer
        if hasattr(model, 'optimizer'):
            return model.optimizer.param_groups[0]['lr']
        return 0.0
    
    def _compute_layer_sparsity(self, model: torch.nn.Module) -> Dict[str, float]:
        layer_sparsity = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                sparsity = (grad.abs() < 1e-6).float().mean().item()
                layer_sparsity[name] = sparsity
        return layer_sparsity
    
    def _compute_effective_compression(self, sparsity: float) -> float:
        # Consider overhead of storing indices
        index_overhead = 0.2  # 20% overhead for storing indices
        return 1 / (1 - sparsity + index_overhead)
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of all tracked metrics."""
        return {
            'energy': {
                'total_energy': sum(e.total_energy for e in self.energy_history),
                'avg_compute_energy': np.mean([e.compute_energy for e in self.energy_history]),
                'avg_memory_energy': np.mean([e.memory_energy for e in self.energy_history]),
                'avg_communication_energy': np.mean([e.communication_energy for e in self.energy_history])
            },
            'memory': {
                'avg_peak_memory': np.mean([m['peak_memory'] for m in self.memory_profile]),
                'max_peak_memory': max(m['peak_memory'] for m in self.memory_profile)
            },
            'communication': {
                'total_bytes_sent': sum(c['bytes_sent'] for c in self.communication_stats),
                'avg_compression_ratio': np.mean([c['compression_ratio'] for c in self.communication_stats])
            },
            'convergence': {
                'final_accuracy': self.convergence_info[-1]['accuracy'] if self.convergence_info else None,
                'accuracy_history': [c['accuracy'] for c in self.convergence_info]
            },
            'sparsity': {
                'avg_sparsity': np.mean([s['global_sparsity'] for s in self.sparsity_stats]),
                'effective_compression': np.mean([s['effective_compression'] for s in self.sparsity_stats])
            }
        } 