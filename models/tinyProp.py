import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.nn.common_types import _size_2_t

class TinyPropParams:
    
    def __init__(self, S_min=0.1, S_max=0.9, zeta=1.0, number_of_layers=5, random_skip=False):
        # Phi parameters
        self.phi_k = 0.5  # Initial phi_k value
        self.phi_max = 0.99  # Maximum phi_k value
        self.phi_momentum = 0.8  # Momentum for phi_k updates
        self.phi_increase_factor = 1.05  # How much to increase phi_k when loss increases
        self.phi_decrease_factor = 0.9  # How much to decrease phi_k when loss decreases
        self.min_phi = 0.3  # Minimum phi_k value
        self.phi_k_decay_rate = 0.95  # Rate at which phi_k decays during loss spikes
        
        # Dynamic phi_max parameters
        self.initial_phi_max = 0.99  # Initial maximum phi_k value
        self.final_phi_max = 0.999  # Final maximum phi_k value
        self.phi_max_growth_rate = 0.001  # How much to increase phi_max per round
        self.phi_max_rounds = 100  # Number of rounds to reach final_phi_max
        self.current_phi_max = self.initial_phi_max  # Current phi_max value
        
        # Loss tracking parameters
        self.loss_window_size = 5  # Number of batches to consider for loss history
        self.loss_threshold = 0.2  # Initial loss change threshold
        self.min_loss_threshold = 0.01  # Minimum loss change threshold
        self.loss_threshold_decay = 0.97  # How much to decay the loss threshold per round
        self.loss_spike_threshold = 1.5  # Threshold for detecting loss spikes
        self.relative_threshold = 0.01  # Threshold for relative loss change
        
        # Batch skipping parameters (removed warmup)
        self.warmup_rounds = 0  # No warmup, start skipping immediately
        self.min_skip_ratio = 0.2  # Higher initial skip ratio
        self.max_skip_ratio = 0.9  # Higher maximum skip ratio
        self.skip_ratio_growth = 0.1  # How much to increase skip ratio per round
        
        # Compression parameters (removed warmup)
        self.compression_warmup = 0  # No warmup, start compression immediately
        self.zeta_growth = 0.05  # How much to increase zeta per round
        self.max_zeta = 2.0  # Higher maximum zeta value
        self.S_min_decay = 0.95  # How much to decrease S_min per round
        self.S_min = S_min  # Minimum sparsity
        self.S_max = S_max  # Maximum sparsity
        self.zeta = zeta  # Initial zeta value
        self.number_of_layers = number_of_layers  # Number of layers in the model
        self.random_skip = random_skip  # Whether to use random skipping
        
        # Debug and tracking
        self.debug_frequency = 5  # More frequent debug output
        self.metrics_file = "client_metrics.csv"  # File to store client metrics
        
        # Dynamic threshold parameters
        self.max_phi = 0.99  # Higher max phi
        self.prev_grad_norm = 0.0  # Previous gradient norm for phi computation
        self.loss_history = []
        self.loss_threshold_decay_rate = 0.95  # Slower decay rate
        
        # Adaptive phi_k parameters
        self.phi_k_restore_rate = 0.1  # Faster restoration rate
        self.target_phi_k = 0.9  # Higher target phi_k
        
        # New parameters for loss spike detection and phi_k decay
        self.high_loss_spike_threshold = 0.8  # Higher threshold for loss spikes
        self.phi_k_decay_factor = 0.95  # Decay factor for phi_k when loss spikes
        
        # New parameters for dynamic sparsity
        self.base_sparsity = 0.1  # Higher base sparsity
        self.sparsity_scaling_factor = 0.3  # Scaling factor for sparsity updates
        self.threshold_scaling_factor = 0.9  # Scaling factor for loss threshold
        self.phi_restore = 0.8  # Target value for phi_k restoration

class TinyPropLayer:
 
    def __init__(self, layerPosition: int):
        self.layerPosition = layerPosition
        self.Y_max = 1e-8
        self.miniBatchBpr = 0.0
        self.miniBatchK = 0.0
        self.epochBpr = []
        self.epochK = []
        self.adaptive_ratio = 1.0
        self.current_round = 0  # Initialize current_round
        self.learning_rate = 0.1  # Initialize learning rate for phi updates
        self.momentum = 0.8  # Initialize momentum for phi updates
        self.phi_min = 0.1  # Initialize minimum phi value
        self.phi_k = 0.3  # Initialize phi_k
        self.last_loss = None  # Initialize last_loss
        self.stats = {
            "skipped_batches": 0,
            "total_batches": 0,
            "phi_k_history": [],
            "loss_change_history": [],
            "loss_history": [],  # Initialize loss history
            "last_batch_loss": None,
            "loss_threshold": 0.01  # Initial loss change threshold
        }

    def reset_batch_stats(self) -> None:
        """Reset batch-level statistics for a new round."""
        self.stats["phi_k_history"] = []
        self.stats["loss_change_history"] = []
        self.stats["skipped_batches"] = 0
        self.last_batch_loss = None

    def BPR(self, params: TinyPropParams, Y: torch.Tensor) -> torch.Tensor:
        # Ensure minimum gradient retention
        min_retention = max(params.S_min, 0.1)  # At least 10% of gradients
        max_retention = min(params.S_max, 0.5)  # At most 50% of gradients
        # Add safeguard against Y_max being too small
        safe_Y_max = max(self.Y_max, 1e-8)
        return (min_retention + Y * (max_retention - min_retention) / safe_Y_max) * (params.zeta ** self.layerPosition)

    def selectGradients(self, grad_output: torch.Tensor, params: TinyPropParams):
        if grad_output.size(1) == 0:
            return torch.empty((2, 0), dtype=torch.int64, device=grad_output.device), torch.empty((0,), dtype=grad_output.dtype, device=grad_output.device)
        
        ratio_from_client = getattr(self, 'adaptive_ratio', 1.0)
        
        # Compute gradient importance with better normalization
        Y = grad_output.abs().sum(dim=1)
        max_Y = torch.max(Y)
        
        # Only update Y_max if the new value is significant
        if max_Y > 1e-8:
            self.Y_max = max(max_Y.item(), 1e-8)  # Ensure Y_max never goes below 1e-8
        
        # Ensure minimum gradient retention
        bpr = self.BPR(params, Y)
        bpr = bpr * ratio_from_client
        bpr = torch.clamp(bpr, 0.1, 0.5)  # Force minimum 10% retention
        
        # Compute number of gradients to keep
        K = torch.round(grad_output.size(1) * bpr)
        K = K.clamp(min=1, max=grad_output.size(1))
        
        # Update statistics
        self.miniBatchBpr += torch.mean(bpr).item()
        self.miniBatchK += torch.mean(K.float()).item()
        K = K.to(torch.int64)
        
        # Select gradients with top-k sparsity and quantization
        idx_list = []
        val_list = []
        scale_list = []  # Store scale factors for each batch
        for batch, k in enumerate(K):
            grad = grad_output[batch].view(-1)
            if grad.numel() == 0:
                continue
            
            k = min(k.item(), grad.numel())
            if k == 0:
                continue
            
            try:
                # Get top-k values and indices
                values, indices = grad.abs().topk(k)
                
                # Apply quantization to the selected gradients
                # Use 8-bit quantization (256 levels)
                max_val = torch.max(torch.abs(values))
                scale = 127.0 / max_val if max_val > 0 else 1.0
                quantized_values = torch.round(values * scale)
                
                # Store scale factor for this batch
                scale_list.append(scale)
                
                # Restore original signs
                quantized_values = quantized_values * torch.sign(grad[indices])
                
                batch_idx = torch.full_like(indices, batch)
                idx_list.append(torch.vstack((batch_idx, indices)))
                val_list.append(quantized_values)
                
                # Track quantization error
                if not hasattr(self, 'quantization_error'):
                    self.quantization_error = []
                dequantized_values = quantized_values / scale
                error = torch.mean(torch.abs(values - dequantized_values))
                self.quantization_error.append(error.item())
                
            except RuntimeError:
                continue
        
        if not idx_list:
            return torch.empty((2, 0), dtype=torch.long, device=grad_output.device), torch.empty((0,), dtype=grad_output.dtype, device=grad_output.device)
        
        indices_sparse = torch.hstack(idx_list)
        values_sparse = torch.cat(val_list)
        
        # Store scale factors in stats
        if not hasattr(self, 'stats'):
            self.stats = {}
        self.stats['scale_factors'] = scale_list
        
        return indices_sparse, values_sparse

    def update_phi(self, params, loss, batch_idx):
        """Update phi_k based on loss change and momentum."""
        if batch_idx == 0:
            self.last_loss = loss
            return
            
        # Calculate loss change and relative loss change
        loss_change = loss - self.last_loss
        relative_loss_change = abs(loss_change) / max(abs(self.last_loss), 1e-8)
        
        # Calculate phi update - larger loss changes should result in larger updates
        # Scale the update based on the direction of loss change
        if loss_change < 0:  # Loss decreased
            phi_update = self.learning_rate * (1 - relative_loss_change)
        else:  # Loss increased
            phi_update = -self.learning_rate * relative_loss_change
        
        # Apply momentum to the update with a more gradual approach
        new_phi = self.phi_k + phi_update * self.momentum
        
        # Clamp phi to valid range with more gradual changes
        new_phi = max(self.phi_min, min(1.0, new_phi))
        
        # Update phi_k and last_loss
        self.phi_k = new_phi
        self.last_loss = loss
        
        # Track phi_k history
        if not hasattr(self, 'phi_k_history'):
            self.phi_k_history = []
        self.phi_k_history.append(self.phi_k)
        
        # Debug logging
        if batch_idx % 5 == 0:
            print(f"[Debug][Batch {batch_idx}] Loss: {loss:.4f}")
            print(f"[Debug][Batch {batch_idx}] Loss Change: {loss_change:.4f}")
            print(f"[Debug][Batch {batch_idx}] Relative Loss Change: {relative_loss_change:.4f}")
            print(f"[Debug][Batch {batch_idx}] Phi Update: {phi_update:.4f}")
            print(f"[Debug][Batch {batch_idx}] New phi_k: {self.phi_k:.4f}")
            print(f"[Debug] phi_k_history: {self.phi_k_history}")

    def should_skip_batch(self, loss: float, params: TinyPropParams) -> bool:
        """Determine if the current batch should be skipped based on loss and phi_k."""
        # Initialize first batch
        if not hasattr(self, 'last_loss') or self.last_loss is None:
            self.last_loss = max(loss, 1e-8)  # Ensure non-zero initial loss
            return False
            
        # Calculate loss change with safeguards
        last_loss = max(self.last_loss, 1e-8)  # Ensure non-zero last loss
        current_loss = max(loss, 1e-8)  # Ensure non-zero current loss
        loss_change = abs(current_loss - last_loss)
        
        # Calculate relative loss change with safeguards
        relative_loss_change = loss_change / last_loss
        
        # Update dynamic phi_max based on current round
        current_round = getattr(self, 'current_round', 0)
        if current_round < params.phi_max_rounds:
            params.current_phi_max = min(
                params.final_phi_max,
                params.initial_phi_max + current_round * params.phi_max_growth_rate
            )
        
        # Calculate dynamic skip ratio (no warmup)
        skip_ratio = min(
            params.max_skip_ratio,
            params.min_skip_ratio + current_round * params.skip_ratio_growth
        )
        
        # Calculate dynamic loss threshold with safeguards
        if len(self.stats["loss_history"]) >= params.loss_window_size:
            avg_loss = sum(self.stats["loss_history"]) / len(self.stats["loss_history"])
            avg_loss = max(avg_loss, 1e-8)  # Ensure non-zero average loss
            dynamic_threshold = max(
                params.min_loss_threshold,
                avg_loss * params.loss_threshold_decay
            )
        else:
            dynamic_threshold = params.loss_threshold
        
        # Progressive compression adjustments (no warmup)
        # Increase zeta over time
        effective_zeta = min(
            params.max_zeta,
            params.zeta * (1 + current_round * params.zeta_growth)
        )
        # Decrease S_min over time
        effective_S_min = params.S_min * (params.S_min_decay ** current_round)
        
        # Calculate layer sparsity based on effective parameters
        layer_sparsity = max(effective_S_min, min(params.S_max, effective_zeta * self.phi_k))
        
        # Determine if batch should be skipped
        should_skip = (
            self.phi_k < params.current_phi_max and  # Use dynamic phi_max
            (
                loss_change < dynamic_threshold or  # Absolute loss change threshold
                relative_loss_change < params.relative_threshold  # Relative loss change threshold
            )
        )
        
        # Enforce minimum skip ratio
        if should_skip:
            current_skip_ratio = self.stats.get("skipped_batches", 0) / (self.stats.get("total_batches", 1) + 1)
            if current_skip_ratio >= skip_ratio:
                should_skip = False
        
        # Update batch statistics
        self.stats["total_batches"] = self.stats.get("total_batches", 0) + 1
        if should_skip:
            self.stats["skipped_batches"] = self.stats.get("skipped_batches", 0) + 1
        
        # Update loss history
        self.stats["loss_history"].append(current_loss)
        if len(self.stats["loss_history"]) > params.loss_window_size:
            self.stats["loss_history"].pop(0)
        
        # Update last_loss for next batch
        self.last_loss = current_loss
        
        # Debug information
        if self.stats["total_batches"] % params.debug_frequency == 0:
            print(f"\n[Batch {self.stats['total_batches']}] Debug Info:")
            print(f"  - Loss: {current_loss:.4f}")
            print(f"  - Loss Change: {loss_change:.4f}")
            print(f"  - Relative Loss Change: {relative_loss_change:.4f}")
            print(f"  - Dynamic Threshold: {dynamic_threshold:.4f}")
            print(f"  - Current phi_k: {self.phi_k:.4f}")
            print(f"  - Current phi_max: {params.current_phi_max:.4f}")
            print(f"  - Layer Sparsity: {layer_sparsity:.4f}")
            print(f"  - Skip Ratio: {skip_ratio:.4f}")
            print(f"  - Current Skip Ratio: {self.stats.get('skipped_batches', 0) / self.stats['total_batches']:.4f}")
            print(f"  - Should Skip: {should_skip}")
        
        return should_skip

    def adjust_loss_threshold(self, current_round: int, total_rounds: int) -> None:
        """Adjust the loss threshold based on the current round."""
        self.current_round = current_round
        # Progressive loss threshold that decreases over rounds
        self.stats["loss_threshold"] = max(0.1 * (0.97 ** current_round), 0.005)
        print(f"[Debug] Round {current_round} loss threshold: {self.stats['loss_threshold']:.4f}")

    def apply_gradient_sparsification(self, model: nn.Module, params: TinyPropParams) -> None:
        """Apply gradient sparsification to the model."""
        with torch.no_grad():
            total_params = 0
            total_sparse_params = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Calculate layer sparsity based on effective parameters
                    if self.current_round >= params.compression_warmup:
                        # Increase zeta over time
                        effective_zeta = min(
                            params.max_zeta,
                            params.zeta * (1 + (self.current_round - params.compression_warmup) * params.zeta_growth)
                        )
                        # Decrease S_min over time
                        effective_S_min = params.S_min * (params.S_min_decay ** (self.current_round - params.compression_warmup))
                    else:
                        effective_zeta = params.zeta
                        effective_S_min = params.S_min
                    
                    # Calculate layer sparsity
                    layer_sparsity = max(effective_S_min, min(params.S_max, effective_zeta * params.phi_k))
                    
                    # Calculate number of parameters to keep
                    num_params = param.numel()
                    num_keep = int(num_params * (1 - layer_sparsity))
                    
                    # Get top-k values and indices
                    grad_flat = param.grad.data.view(-1)
                    values, indices = torch.topk(torch.abs(grad_flat), k=num_keep)
                    
                    # Create sparse gradient
                    sparse_grad = torch.zeros_like(grad_flat)
                    sparse_grad[indices] = grad_flat[indices]
                    
                    # Update gradient
                    param.grad.data = sparse_grad.view(param.grad.data.shape)
                    
                    # Update compression statistics
                    total_params += num_params
                    total_sparse_params += num_keep
            
            # Update compression ratio
            if total_params > 0:
                self.stats["compression_ratio"] = 1.0 - (total_sparse_params / total_params)

class SparseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, tpParams: TinyPropParams, tpInfo: TinyPropLayer, bias=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.tpParams = tpParams
        ctx.tpInfo = tpInfo
        return F.linear(input, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        indices, values = ctx.tpInfo.selectGradients(grad_output, ctx.tpParams)
        sparse_grad = torch.sparse_coo_tensor(indices, values, grad_output.size(), device=grad_output.device)
        if ctx.needs_input_grad[0]:
            grad_input = torch.sparse.mm(sparse_grad, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.sparse.mm(sparse_grad.t(), input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)  # Sum over batch dimension
        return grad_input, grad_weight, None, None, grad_bias

class SparseConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, padding_mode, _reversed_padding_repeated_twice, tpParams: TinyPropParams, tpInfo: TinyPropLayer):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.tpParams = tpParams
        ctx.tpInfo = tpInfo
        
        if padding_mode != 'zeros':
            padded_input = F.pad(input, _reversed_padding_repeated_twice, mode=padding_mode)
            return F.conv2d(padded_input, weight, bias, stride, 0, dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        flattened = grad_output.flatten(start_dim=1)
        indices, values = ctx.tpInfo.selectGradients(flattened, ctx.tpParams)
        sparse_flat = torch.zeros_like(flattened)
        sparse_flat[indices[0], indices[1]] = values
        grad_output_masked = sparse_flat.view_as(grad_output).to(weight.device)
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape,
                weight,
                grad_output_masked,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input,
                weight.shape,
                grad_output_masked,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])  # Sum over batch and spatial dimensions
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None

class SparseConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, padding_mode, _reversed_padding_repeated_twice, tpParams: TinyPropParams, tpInfo: TinyPropLayer):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.tpParams = tpParams
        ctx.tpInfo = tpInfo
        
        if padding_mode != 'zeros':
            padded_input = F.pad(input, _reversed_padding_repeated_twice, mode=padding_mode)
            return F.conv1d(padded_input, weight, bias, stride, 0, dilation, groups)
        return F.conv1d(input, weight, bias, stride, padding, dilation, groups)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        flattened = grad_output.flatten(start_dim=1)
        indices, values = ctx.tpInfo.selectGradients(flattened, ctx.tpParams)
        sparse_flat = torch.zeros_like(flattened)
        sparse_flat[indices[0], indices[1]] = values
        grad_output_masked = sparse_flat.view_as(grad_output).to(weight.device)
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv1d_input(
                input.shape,
                weight,
                grad_output_masked,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv1d_weight(
                input,
                weight.shape,
                grad_output_masked,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2])  # Sum over batch and spatial dimensions
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None

class TinyPropLinear(TinyPropLayer, nn.Linear):
    def __init__(self, in_features: int, out_features: int, tinyPropParams: TinyPropParams, layer_number: int, bias: bool=True):
        TinyPropLayer.__init__(self, tinyPropParams.number_of_layers - layer_number)
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.tpParams = tinyPropParams

    def forward(self, input):
        return SparseLinear.apply(input, self.weight, self.tpParams, self, self.bias)

class TinyPropConv2d(TinyPropLayer, nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 tinyPropParams: TinyPropParams,
                 layer_number: int,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'):
        TinyPropLayer.__init__(self, tinyPropParams.number_of_layers - layer_number)
        
        # Then initialize nn.Conv2d
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=bias, padding_mode=padding_mode)
        self.tpParams = tinyPropParams

    def forward(self, input):
        
        
        return SparseConv2d.apply(
            input, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups, self.padding_mode,
            self._reversed_padding_repeated_twice, self.tpParams, self
        )

class TinyPropConv1d(TinyPropLayer, nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, tinyPropParams=None, layer_number=1):
        TinyPropLayer.__init__(self, tinyPropParams.number_of_layers - layer_number)
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                          dilation, groups, bias)
        self.tpParams = tinyPropParams

    def forward(self, input):
        return SparseConv1d.apply(
            input, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups, self.padding_mode,
            self._reversed_padding_repeated_twice, self.tpParams, self
        )

def get_phi_k(model):
    """Safely get the latest phi value from a model's TinyPropLayer."""
    hist = model.tpLayer.stats.get("phi_k_history", [])
    return hist[-1] if hist else 0.0
