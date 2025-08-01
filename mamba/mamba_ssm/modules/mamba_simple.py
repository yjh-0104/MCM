import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None

try:
    from mamba.mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        nframes=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.nframes = nframes

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        A_s = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_s_log = torch.log(A_s)  # Keep A_s_log in fp32
        self.A_s_log = nn.Parameter(A_s_log)
        self.A_s_log._no_weight_decay = True 

        self.conv1d_s = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_s = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_s = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_s = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_s._no_weight_decay = True

        A_sb = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_sb_log = torch.log(A_sb)  # Keep A_sb_log in fp32
        self.A_sb_log = nn.Parameter(A_sb_log)
        self.A_sb_log._no_weight_decay = True 

        self.conv1d_sb = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_sb = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_sb = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_sb = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_sb._no_weight_decay = True


        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A_s = -torch.exp(self.A_s_log.float())
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v3":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                A_s = -torch.exp(self.A_s_log.float())

                xz_s = xz.chunk(self.nframes, dim=-1)
                xz_s = torch.stack(xz_s,dim=-1)
                xz_s = xz_s.flatten(-2)
                out_s = mamba_inner_fn_no_out_proj(
                    xz_s,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )
                out_s = out_s.reshape(batch,self.d_inner,seqlen//self.nframes,self.nframes).permute(0,1,3,2).flatten(-2)

                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                out = F.linear(rearrange(out + out_b.flip([-1]) + out_s, "b d l -> b l d") / 3, self.out_proj.weight, self.out_proj.bias)
            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type == "v1":
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

                out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)


            elif self.bimamba_type == "v4":
                xz_s = xz.chunk(self.nframes, dim=-1) 
                xz_s = torch.stack(xz_s, dim=-1)
                xz_s = xz_s.flatten(-2)
                
                out_s = mamba_inner_fn_no_out_proj(
                    xz_s,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )
                
                out_s = out_s.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_s = out_s.permute(0, 1, 3, 2).flatten(-2)
            
                out = F.linear(rearrange(out_s, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            
            elif self.bimamba_type == "v5":
                A_sb = -torch.exp(self.A_sb_log.float())
                xz_s = xz.chunk(self.nframes, dim=-1) 
                xz_s = torch.stack(xz_s, dim=-1) 
                xz_s = xz_s.flatten(-2)
                
                out_s = mamba_inner_fn_no_out_proj(
                    xz_s,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )
                
                out_sb = mamba_inner_fn_no_out_proj(
                    xz_s.flip([-1]),
                    self.conv1d_sb.weight,
                    self.conv1d_sb.bias,
                    self.x_proj_sb.weight,
                    self.dt_proj_sb.weight,
                    A_sb,
                    None,
                    None,
                    self.D_sb.float(),
                    delta_bias=self.dt_proj_sb.bias.float(),
                    delta_softplus=True,
                )                
                out_sb = out_sb.flip([-1])
                
                out_s = out_s.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_s = out_s.permute(0, 1, 3, 2).flatten(-2)
                
                out_sb = out_sb.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_sb = out_sb.permute(0, 1, 3, 2).flatten(-2)
            
                out = F.linear(rearrange(out_s + out_sb, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type == "v6":
                A_s = -torch.exp(self.A_s_log.float())
                xz_s = xz.chunk(self.nframes, dim=-1) 
                xz_s = torch.stack(xz_s, dim=-1) 
            
                xz_sb = xz_s.flip(-2) 
            
                xz_s = xz_s.flatten(-2)
                xz_sb = xz_sb.flatten(-2)
            
                out_s = mamba_inner_fn_no_out_proj(
                    xz_s,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )
            
                A_sb = -torch.exp(self.A_sb_log.float())
                out_sb = mamba_inner_fn_no_out_proj(
                    xz_sb, 
                    self.conv1d_sb.weight,
                    self.conv1d_sb.bias,
                    self.x_proj_sb.weight,
                    self.dt_proj_sb.weight,
                    A_sb,
                    None,
                    None,
                    self.D_sb.float(),
                    delta_bias=self.dt_proj_sb.bias.float(),
                    delta_softplus=True,
                )
                out_sb = out_sb.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_sb = out_sb.permute(0, 1, 3, 2).flip(-1).flatten(-2)
            
                out_s = out_s.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_s = out_s.permute(0, 1, 3, 2).flatten(-2)
            
                out = F.linear(rearrange(out_s + out_sb, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type == "v7":
                A_s = -torch.exp(self.A_s_log.float())
                xz_s = xz.chunk(self.nframes, dim=-1) 
                xz_s = torch.stack(xz_s, dim=-1)  # (B, d_inner, tokens_per_frame, F)
            
                xz_sb = xz_s.flip(-1)
            
                xz_s = xz_s.flatten(-2)
                xz_sb = xz_sb.flatten(-2)
            
                out_s = mamba_inner_fn_no_out_proj(
                    xz_s,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )
            
                A_sb = -torch.exp(self.A_sb_log.float())
                out_sb = mamba_inner_fn_no_out_proj(
                    xz_sb, 
                    self.conv1d_sb.weight,
                    self.conv1d_sb.bias,
                    self.x_proj_sb.weight,
                    self.dt_proj_sb.weight,
                    A_sb,
                    None,
                    None,
                    self.D_sb.float(),
                    delta_bias=self.dt_proj_sb.bias.float(),
                    delta_softplus=True,
                )
            
                out_sb = out_sb.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_sb = out_sb.permute(0, 1, 3, 2).flip(-2).flatten(-2) 
            
                out_s = out_s.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_s = out_s.permute(0, 1, 3, 2).flatten(-2)
            
                out = F.linear(rearrange(out_s + out_sb, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

            elif self.bimamba_type == "v8":
                A_s = -torch.exp(self.A_s_log.float())
                A_sb = -torch.exp(self.A_sb_log.float())
                
                xz_s = xz.chunk(self.nframes, dim=-1)   
                xz_s = torch.stack(xz_s, dim=-1)   
                
                xz_fwd = xz_s.flatten(-2)
                out_fwd = mamba_inner_fn_no_out_proj(
                    xz_fwd,
                    self.conv1d_s.weight,
                    self.conv1d_s.bias,
                    self.x_proj_s.weight,
                    self.dt_proj_s.weight,
                    A_s,
                    None,
                    None,
                    self.D_s.float(),
                    delta_bias=self.dt_proj_s.bias.float(),
                    delta_softplus=True,
                )

                out_fwd = out_fwd.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_fwd = out_fwd.permute(0, 1, 3, 2).flatten(-2)  # (B, d_inner, seqlen)
                
                xz_rev_all = xz_s.flatten(-2).flip(-1)    
                out_rev_all = mamba_inner_fn_no_out_proj(
                    xz_rev_all,
                    self.conv1d_sb.weight,
                    self.conv1d_sb.bias,
                    self.x_proj_sb.weight,
                    self.dt_proj_sb.weight,
                    A_sb,
                    None,
                    None,
                    self.D_sb.float(),
                    delta_bias=self.dt_proj_sb.bias.float(),
                    delta_softplus=True,
                )
                out_rev_all = out_rev_all.flip(-1) 
                out_rev_all = out_rev_all.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_rev_all = out_rev_all.permute(0, 1, 3, 2).flatten(-2)
                
                xz_rev_token = xz_s.flip(-2)                  
                xz_rev_token = xz_rev_token.flatten(-2)       
                out_rev_token = mamba_inner_fn_no_out_proj(
                    xz_rev_token,
                    self.conv1d_sb.weight,
                    self.conv1d_sb.bias,
                    self.x_proj_sb.weight,
                    self.dt_proj_sb.weight,
                    A_sb,
                    None,
                    None,
                    self.D_sb.float(),
                    delta_bias=self.dt_proj_sb.bias.float(),
                    delta_softplus=True,
                )

                out_rev_token = out_rev_token.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_rev_token = out_rev_token.permute(0, 1, 3, 2).flip(-1).flatten(-2)
                
                xz_rev_frame = xz_s.flip(-1)                     
                xz_rev_frame = xz_rev_frame.flatten(-2)        
                out_rev_frame = mamba_inner_fn_no_out_proj(
                    xz_rev_frame,
                    self.conv1d_sb.weight,
                    self.conv1d_sb.bias,
                    self.x_proj_sb.weight,
                    self.dt_proj_sb.weight,
                    A_sb,
                    None,
                    None,
                    self.D_sb.float(),
                    delta_bias=self.dt_proj_sb.bias.float(),
                    delta_softplus=True,
                )

                out_rev_frame = out_rev_frame.reshape(batch, self.d_inner, seqlen // self.nframes, self.nframes)
                out_rev_frame = out_rev_frame.permute(0, 1, 3, 2).flip(-2).flatten(-2)
                
                out_combined = (out_fwd + out_rev_all + out_rev_token + out_rev_frame) / 4

                out = F.linear(rearrange(out_combined, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

            
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)