import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.nn.common_types import _size_2_t

class TinyPropParams:
    
    def __init__(self, S_min: float, S_max: float, zeta: float, number_of_layers: int):
        self.S_min = S_min
        self.S_max = S_max
        self.zeta = zeta
        self.number_of_layers = number_of_layers

class TinyPropLayer:
 
    def __init__(self, layerPosition: int):
        self.layerPosition = layerPosition
        self.Y_max = 1e-8
        self.miniBatchBpr = 0.0
        self.miniBatchK = 0.0
        self.epochBpr = []
        self.epochK = []

    def BPR(self, params: TinyPropParams, Y: torch.Tensor) -> torch.Tensor:
        return (params.S_min + Y * (params.S_max - params.S_min) / self.Y_max) * (params.zeta ** self.layerPosition)

    def selectGradients(self, grad_output: torch.Tensor, params: TinyPropParams):
        if grad_output.size(1) == 0:
            print("Warning: grad_output has zero dimension. Skipping top-k selection.")
        
            empty_indices = torch.empty((2, 0), dtype=torch.int64, device=grad_output.device)
            empty_values = torch.empty((0,), dtype=grad_output.dtype, device=grad_output.device)
            return empty_indices, empty_values
        
        ratio_from_client = getattr(self, 'adaptive_ratio', 1.0) 
        
        Y = grad_output.abs().sum(dim=1)
        max_Y = torch.max(Y)
        if max_Y > self.Y_max:
            self.Y_max = max_Y.item()
            
        bpr = (params.S_min + Y*(params.S_max - params.S_min)/self.Y_max) * (params.zeta ** self.layerPosition)
        bpr = bpr * ratio_from_client
        bpr = torch.clamp(bpr, 0.0, 1.0)
        
        K = torch.round(grad_output.size(1) * bpr)
        K = K.clamp(min=1, max=grad_output.size(1))
        self.miniBatchBpr += torch.mean(bpr).item()
        self.miniBatchK += torch.mean(K.float()).item()
        K = K.to(torch.int64)
        
        idx_list = []
        val_list = []
        for batch, k in enumerate(K):
            grad = grad_output[batch]
            k = min(k.item(), grad.numel())  # extra safe
            if k == 0:
                continue  # skip if zero
            _, indices = grad.abs().topk(k)
            batch_idx = torch.full_like(indices, batch)
            idx_list.append(torch.vstack((batch_idx, indices)))
            val_list.append(torch.index_select(grad, -1, indices))
            


            
        indices_sparse = torch.hstack(idx_list)
        values_sparse = torch.cat(val_list)
        return indices_sparse, values_sparse

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
            grad_bias = grad_output.sum(0)
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
            grad_input = F.conv_transpose2d(grad_output_masked, weight, None, ctx.stride, ctx.padding, groups=ctx.groups, dilation=ctx.dilation)
        if ctx.needs_input_grad[1]:
            permuted_grad = grad_output_masked.permute(1, 0, 2, 3)
            input_channels = torch.unbind(input, dim=1)
            weight_grad_list = []
            for channel in input_channels:
                weight_grad_list.append(F.conv2d(channel, permuted_grad, None, ctx.stride, ctx.padding, groups=ctx.groups, dilation=ctx.dilation))
            grad_weight = torch.stack(weight_grad_list, dim=0).permute(1, 0, 2, 3)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_masked.sum(dim=(0,2,3))
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
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode)
        self.tpParams = tinyPropParams

    def forward(self, input):
        return SparseConv2d.apply(input, self.weight, self.bias, self.stride, self.padding,
                                    self.dilation, self.groups, self.padding_mode, self._reversed_padding_repeated_twice,
                                    self.tpParams, self)
