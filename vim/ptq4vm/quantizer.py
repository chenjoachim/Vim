from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict
REAL_INT8 = False

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lv, size):
        ctx.save_for_backward(torch.Tensor([n_lv, size]))
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        saved, = ctx.saved_tensors
        n_lv, size = int(saved[0]), float(saved[1])

        if n_lv == 0:
            return grad_output, None, None
        else:
            scale = 1 / np.sqrt(n_lv * size)
            return grad_output.mul(scale), None, None

class Q_Linear(nn.Linear):
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.n_lv = 0
        self.qmax = self.n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        self.num = 100
        self.register_buffer("eps", torch.tensor(1e-8))
        self.smoothing = False
        self.real_int8 = False
        self.qbit = 4
        self.int_weight = torch.Tensor(1)
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
    
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if not self.per_channel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def set_real_int8(self):
        self.real_int8 = True
        with torch.no_grad():
            weight = self.weight * self.act_func.smooth_scale
            if self.n_lv == 256:
                self.qbit = 8
                align = 16
            elif self.n_lv == 16:
                self.qbit = 4
                align = 32
            qmax = self.n_lv // 2 - 1
            weight = F.hardtanh(weight / self.s, -qmax, qmax)
            int_weight = torch.round(weight).to(torch.int8).detach()  # (n, k)

            # Pre-pad weight+smooth_scale once so forward() doesn't re-allocate each call.
            # Zero-padding preserves dot products, so w_col_sum and accuracy are unchanged.
            k = int_weight.shape[1]
            pad = (align - k % align) % align
            self._gemm_pad = pad
            if pad > 0:
                int_weight = F.pad(int_weight.float(), (0, pad)).to(torch.int8)
                padded_smooth = F.pad(self.act_func.smooth_scale.detach(), (0, pad)).contiguous()
                self.register_buffer("_padded_smooth_scale", padded_smooth)

            self.w_col_sum = int_weight.float().sum(dim=1)  # (n,)

            if self.qbit == 8:
                self.int_weight = int_weight.contiguous()
            else:
                # Pack two int4 values into one byte: low nibble = w[2i], high nibble = w[2i+1]
                n_rows, new_k = int_weight.shape
                assert new_k % 2 == 0, f"k={new_k} must be even for int4 packing"
                w_flat = (int_weight.view(n_rows, new_k // 2, 2).to(torch.uint8) & 0x0F)
                self.int_weight = (w_flat[:, :, 0] | (w_flat[:, :, 1] << 4)).detach().contiguous()

            # Hoist invariants out of forward(): squeeze/unsqueeze/contiguous on act scale+zero,
            # pre-contiguous the weight scale, and prime the import path for vim_GEMM once.
            self._scale_x_base = self.act_func.s.detach().squeeze().unsqueeze(-1).contiguous()
            self._z_base = self.act_func.z.detach().squeeze().unsqueeze(-1).contiguous()
            self._per_token = self._scale_x_base.numel() > 1
            self._s_contig = self.s.detach().contiguous()
            # Dtype/batch-dependent pieces get built lazily on first forward.
            self._cached_dtype = None
            self._cached_batch = None
            self._cached_scale_x = None
            self._cached_z = None
            self._cached_w_col_sum = None

            import sys, os
            _gemm_path = os.path.join(os.path.dirname(__file__), '..', 'cuda_measure')
            if _gemm_path not in sys.path:
                sys.path.insert(0, _gemm_path)
            import vim_GEMM  # noqa: F401  (warm the import cache)

    def initialize(self, n_lv, per_channel=False, trunc=False):
        x = self.weight * self.act_func.smooth_scale
        
        self.n_lv = n_lv
        self.qmax = n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = per_channel     
        if not trunc:
            if self.per_channel:
                del self.s
                max_val = x.abs().max(dim=1, keepdim=True)[0]
                val = max_val / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val))
                
            else:
                max_val = x.abs().max()
                self.s.data = max_val / self.qmax
        else:
            
            if self.per_channel:
                x = x.flatten(1)
            else:
                x = x.flatten().unsqueeze(0)

            xmin = x.min(1)[0]
            xmax = x.max(1)[0]
            
            if self.per_channel:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            
            xrange = torch.max(xmin.abs(), xmax)
            
            for i in range(1, self.num + 1):
                tmp_max = xrange / self.num * i
                scale = torch.max(tmp_max / self.qmax, self.eps)
                if self.per_channel:
                    scale = scale.reshape(new_shape)
                x_round = torch.round(x/scale)
                x_q = self.quantize_efficient(x_round, scale)
                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(score, best_score)
                
            max_val = torch.max(best_max, torch.zeros_like(best_max))
            if self.per_channel:
                del self.s
                val = torch.max(max_val / self.qmax, self.eps).unsqueeze(1)
                self.register_parameter("s",torch.nn.Parameter(val))
            else:
                self.s.data = torch.max(max_val / self.qmax, self.eps)

        self.smoothing = True
        print("Q_Linear Max s :" +  str(self.s.max()))
 
    def _weight_quant(self): 
        s = self.s 
        if self.smoothing:
            weight = self.weight * self.act_func.smooth_scale
        else:
            weight = self.weight
        weight = F.hardtanh((weight / s), self.qmin, self.qmax)
        weight = RoundQuant.apply(weight) * s
        return weight

    def _weight_int(self):
        with torch.no_grad():        
            weight = F.hardtanh(self.weight / self.s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            weight = torch.round(weight)
        return weight
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
        
        if self.real_int8:
            import vim_GEMM  # already primed in set_real_int8()

            dtype = x.dtype
            if self._cached_dtype is not dtype:
                self._cached_dtype = dtype
                self._cached_z = self._z_base.to(dtype).contiguous()
                self._cached_w_col_sum = self.w_col_sum.to(dtype).contiguous()
                self._cached_batch = None  # bcast tensors depend on cached_z dtype

            if self._per_token:
                batch = x.shape[0]
                if self._cached_batch != batch:
                    self._cached_batch = batch
                    # per-token: kernel indexes over batch*seq_len rows
                    self._cached_scale_x = self._scale_x_base.repeat(batch, 1).contiguous()
                    self._cached_z_bcast = self._cached_z.repeat(batch, 1).contiguous()
                scale_x = self._cached_scale_x
                z = self._cached_z_bcast
            else:
                scale_x = self._scale_x_base
                z = self._cached_z

            # Weight + smooth_scale were pre-padded in set_real_int8(); only x needs per-call padding.
            pad = self._gemm_pad
            if pad > 0:
                x = F.pad(x.contiguous(), (0, pad))
                smooth_scale = self._padded_smooth_scale
            else:
                smooth_scale = self.act_func.smooth_scale

            result = vim_GEMM.vim_GEMM(
                x.contiguous(),
                self.int_weight,
                smooth_scale,
                scale_x,
                self._s_contig,
                z,
                self._cached_w_col_sum,
                16,
                self.qbit,
            )

            if self.bias is not None:
                result = result + self.bias

            return result
        elif self.n_lv == 0:    
            if self.smoothing:
                weight = self.weight * self.act_func.smooth_scale
            else:
                weight = self.weight
            return F.linear(x, weight, self.bias)
        else:
            try:
                weight = self._weight_quant()
            except Exception as e:
                raise RuntimeError(
                    f"Weight quantization failed: weight shape={tuple(self.weight.shape)}"
                ) from e
            return F.linear(x, weight, self.bias)

class Q_Act(nn.Module):
    def __init__(self, symmetric=False):
        super(Q_Act, self).__init__()
        # n_lv, qmax, qmin -> refer initialize() function
        self.n_lv = 0
        self.qmax = 0
        self.qmin = 0
        self.symmetric = symmetric
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        self.num = 100
        self.register_buffer("eps", torch.tensor(1e-8))
        self.per_token = False
        self.smooth_scale = Parameter(torch.ones(1))
        self.register_buffer("z", torch.tensor(0.0))
        self.smoothing = False
        self.real_int8 = False
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
    
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.per_token:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def set_real_int8(self):
        self.real_int8 = True 

    def initialize(self, n_lv, tensor, per_token=False, trunc=False):
        x = tensor / self.smooth_scale
        self.n_lv = n_lv
        self.per_token = per_token

        if self.symmetric:
            self.qmax = n_lv // 2 - 1
            self.qmin = -self.qmax
            self.register_buffer("z", torch.tensor(0.0))
        else:
            self.qmax = n_lv - 1
            self.qmin = 0

        if not trunc:
            if self.per_token:
                b,l,d = x.shape
                x = x.permute(0,2,1) # b, d, l
                x = x.reshape(-1, l) # bd, l
                x = x.permute(1,0) # l, bd
                del self.s
                if self.symmetric:
                    max_val = x.abs().max(dim=1, keepdim=True)[0]
                    val = max_val / self.qmax
                else:
                    max_val = x.max(dim=1, keepdim=True)[0]
                    min_val = x.min(dim=1, keepdim=True)[0]
                    val = (max_val - min_val) / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val.unsqueeze(0)))
                if not self.symmetric:
                    self.register_buffer("z", torch.round(-min_val.unsqueeze(0) / self.s))

            else:
                if self.symmetric:
                    max_val = x.abs().max()
                    self.s.data = max_val / self.qmax
                else:
                    max_val = x.max()
                    min_val = x.min()
                    val = (max_val - min_val) / self.qmax
                    self.s.data = torch.tensor(val)
                    self.register_buffer("z", torch.round(-min_val / self.s))
        else:
            if self.per_token:
                b,l,d = x.shape
                x = x.permute(0,2,1) # b, d, l
                x = x.reshape(-1, l) # bd, l
                x = x.permute(1,0)
            else:
                x = x.flatten().unsqueeze(0)
            
            xmin = x.min(1)[0]
            xmax = x.max(1)[0]
            
            if self.per_token:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            
            for i in range(1, self.num + 1):
                alpha = i / self.num
                tmp_min = xmin * (1-alpha) + xmax * alpha
                tmp_max = xmin * alpha + xmax * (1-alpha)         
                
                scale = (tmp_max - tmp_min) / (self.qmax - self.qmin)            
                scale = torch.max((tmp_max - tmp_min) / (self.qmax - self.qmin), self.eps)
                zero = torch.round(-tmp_min / scale) + self.qmin
                
                # Reshape for per-channel quantization if needed
                if self.per_token:
                    scale = scale.reshape(new_shape)
                    zero = zero.reshape(new_shape)

                # Perform quantization with the computed scale and zero point
                x_round = torch.round(x / scale)
                x_q = self.quantize_efficient(x_round, scale, zero)

                # Compute score and update best values
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, tmp_min, best_min)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(best_score, score)

            # Final scale and zero point calculation
            min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))

            max_val = torch.max(best_max, torch.zeros_like(best_max))
            if self.per_token:
                del self.s
                val = torch.max((max_val_pos - min_val_neg) / (self.qmax - self.qmin), self.eps).unsqueeze(1).unsqueeze(0)
                self.register_parameter("s",torch.nn.Parameter(val))
                self.register_buffer("z", torch.clamp(self.qmin - torch.round(min_val_neg.unsqueeze(1).unsqueeze(0) / self.s), self.qmin, self.qmax))
            else:
                self.s.data = torch.max((max_val_pos - min_val_neg) / (self.qmax - self.qmin), self.eps)
                self.register_buffer("z", torch.clamp(self.qmin - torch.round(min_val_neg / self.s), self.qmin, self.qmax))
        self.smoothing = True
        print("Q_Act Max s :" +  str(self.s.max())) 

        
    def forward(self, x):
        if self.real_int8: # Kernel includes act quant procedure
            return x
        if self.n_lv == 0:
            if self.smoothing:
                return x / self.smooth_scale
            else:
                return x
        else:
            if self.smoothing:
                # s = self.s
                x = x / self.smooth_scale
                if self.per_token:

                    # max_val = x.max(dim=-1, keepdim=True)[0]
                    # min_val = x.min(dim=-1, keepdim=True)[0]
                    # s = (max_val - min_val) / self.qmax
                    # z = -(min_val / s).round()
                    s = self.s
                    z = self.z

                else:
                    s = self.s
                    z = self.z
                x = F.hardtanh(x / s + z, self.qmin, self.qmax)
                x = RoundQuant.apply(x - z) * s
                
            else:
                s = self.s
                z = self.z
                x = F.hardtanh(x / s + z, self.qmin, self.qmax)
                x = RoundQuant.apply(x - z) * s
            return x

            
def initialize(layer, input, residual, n_lvw, n_lva, act=False, weight=False, per_channel=False, per_token=False, trunc=False):    
    def initialize_hook(module, input, output): 
        if isinstance(module, (Q_Linear)) and weight:
            module.initialize(n_lvw, per_channel=per_channel, trunc=trunc)
        if isinstance(module, (Q_Act)):
            module.initialize(n_lva, input[0], per_token=per_token, trunc=trunc)

            
    hooks = []
    for name, module in layer.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    with torch.no_grad():
        input = input.to('cuda')
        # residual = residual.to('cuda')
        if isinstance(layer, nn.DataParallel):
            output = layer.module(input, residual)
        else:
            output = layer(input, residual)
            
    for hook in hooks:
        hook.remove()
        
class QuantOps(object):
    initialize = initialize
    Act = Q_Act
    Linear = Q_Linear
