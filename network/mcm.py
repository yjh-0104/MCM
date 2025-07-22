from __future__ import annotations

import torch.nn as nn
import torch 
from mamba.mamba_ssm.modules.mamba_simple import Mamba
import torch.nn.functional as F 
from timm.models.layers import DropPath, trunc_normal_
import math

class DWConv(nn.Module):
    """3D depthwise convolution layer for patch embeddings."""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        """
        Args:
            x: input tensor of shape (B, N, C)
            nf, H, W: dimensions for reshaping
        Returns:
            Output tensor after depthwise conv and flattening
        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MambaMlp(nn.Module):
    """MLP block with depthwise conv for Mamba layer."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        """
        Args:
            x: input tensor
            nf, H, W: dimensions for reshaping
        Returns:
            Output tensor after MLP with depthwise conv
        """
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MambaBlock(nn.Module):
    """Single Mamba block with normalization, Mamba, and MLP."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type="v5",
                nframes=5,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MambaMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        Args:
            x: input tensor (B, C, nf, H, W)
        Returns:
            Output tensor after Mamba block
        """
        B, C, nf, H, W = x.shape
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class PatchEmbeddings(nn.Module):
    """Patch embedding using 2D convolution."""
    def __init__(self, patch_size, stride, in_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: input image (B, C, H, W)
        Returns:
            embeddings: patch embeddings
            height, width: spatial dimensions
        """
        embeddings = self.proj(x) 
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
        embeddings = self.layer_norm(embeddings) 
        return embeddings, height, width

class MambaEncoder(nn.Module):
    """Stacked Mamba blocks for multi-stage feature extraction."""
    def __init__(self, in_chans=1, depths=None, dims=None,
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        patch_sizes = [7, 3, 3, 3] 
        strides = [4, 2, 2, 2] 
        in_channels = [2, 64, 128, 256]
        hidden_sizes = [64, 128, 256, 512] 

        self.patch_embeddings = nn.ModuleList([
            PatchEmbeddings(patch_sizes[i], strides[i], in_channels[i], hidden_sizes[i])
            for i in range(len(hidden_sizes))
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in [64, 128, 256, 512] 
        ])
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[nn.Sequential(
                MambaBlock(dim=dims[i], drop_path=dp_rates[i])
                    ) for j in range(depths[i])]
            )
            self.stages.append(stage)
        self.out_indices = out_indices

    def forward_features(self, x):
        outs = []
        bz, nf, nc, h, w = x.shape
        x = x.reshape(bz*nf, x.shape[-3], x.shape[-2], x.shape[-1])
        hs = x
        for idx, x in enumerate(zip(self.patch_embeddings, self.layer_norms, self.stages)):
            embedding_layer, norm_layer, mam_stage = x
            hs, height, width = embedding_layer(hs)
            hs = norm_layer(hs)
            hs = hs.reshape(bz*nf, height, width, -1).permute(0, 3, 1, 2).contiguous()
            hs = hs.reshape(bz, nf, hs.shape[-3], hs.shape[-2], hs.shape[-1]).transpose(1, 2)
            hs = mam_stage(hs)
            hs = hs.transpose(1, 2)
            hs = hs.reshape(bz*nf, hs.shape[-3], hs.shape[-2], hs.shape[-1])
            outs.append(hs)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class UpBlock(nn.Module):
    """Decoder upsampling block with skip connection and convs."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        """
        Args:
            x: input tensor (bz, nf, C, H, W)
            skip: skip connection tensor
        Returns:
            Output tensor after upsampling and conv
        """
        bz, nf, C, H, W = x.shape
        x_reshaped = x.view(bz * nf, C, H, W)
        target_H, target_W = skip.shape[3], skip.shape[4]
        x_up = F.interpolate(x_reshaped, size=(target_H, target_W), mode='bilinear', align_corners=False)
        x_up = x_up.view(bz, nf, C, target_H, target_W)
        concat = torch.cat([x_up, skip], dim=2)
        concat = concat.view(bz * nf, concat.shape[2], target_H, target_W)
        out = self.conv(concat)
        out = out.view(bz, nf, out.shape[1], target_H, target_W)
        return out

class DecoderWithDFH(nn.Module):
    """Decoder with Dual-Path Fusion Head."""
    def __init__(self, encoder_channels, decoder_channels, nf, final_channels=2):
        super().__init__()
        self.nf = nf
        self.bottleneck_channels = encoder_channels[-1]
        self.up_blocks = nn.ModuleList()
        in_ch = self.bottleneck_channels
        num_up = len(decoder_channels)
        for i in range(num_up):
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = decoder_channels[i]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=4,
            stride=4,
            padding=0
        )
        self.forward_temporal = nn.Conv3d(in_ch, in_ch, kernel_size=(3,1,1), padding=(1,0,0), groups=in_ch, bias=False)
        self.backward_temporal = nn.Conv3d(in_ch, in_ch, kernel_size=(3,1,1), padding=(1,0,0), groups=in_ch, bias=False)
        self.pointwise_temporal = nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False)
        self.out_conv = nn.Conv2d(in_ch, final_channels, kernel_size=1)

    def forward(self, encoder_outputs, target_h, target_w, bz, nf):
        """
        Args:
            encoder_outputs: multi-scale features from encoder
            target_h, target_w: target output spatial size
            bz, nf: batch size and frame number
        Returns:
            Output deformation field
        """
        skips = []
        for feat in encoder_outputs:
            C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
            skips.append(feat.view(bz, nf, C, H, W))
        x = skips[-1]
        num_up = len(self.up_blocks)
        for i, up in enumerate(self.up_blocks):
            skip = skips[-(i + 2)]
            x = up(x, skip)
        bz_nf, C, H, W = x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        x_reshaped = x.view(bz * nf, C, H, W)
        x_upsampled = self.upsample(x_reshaped)
        x = x_upsampled.view(bz, nf, C, target_h, target_w)

        # DFH
        x_perm = x.permute(0, 2, 1, 3, 4)
        x_fwd = self.forward_temporal(x_perm)
        x_bwd = self.backward_temporal(torch.flip(x_perm, dims=[2]))
        x_temporal = (x_fwd + x_bwd) / 2
        x_pw = self.pointwise_temporal(x_temporal)
        mid_idx = nf // 2
        fused = x_pw[:, :, mid_idx:mid_idx+1, :, :]
        fused = fused.squeeze(2)
        out = self.out_conv(fused)
        out = out.unsqueeze(1)
        return out

class SpatialTransformer(nn.Module):
    """Spatial Transformer for image warping."""
    def __init__(self, size):
        super().__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode):
        """
        Args:
            src: source image (B, T, C, H, W)
            flow: deformation field
            mode: interpolation mode
        Returns:
            Warped output image
        """
        B, T, C, H, W = src.size()
        src = src.view(B * T, C, H, W)
        B1, T1, C1, H1, W1 = flow.size()
        flow = flow.view(B1 * T1, C1, H1, W1)
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        out = F.grid_sample(src, new_locs, align_corners=True, mode=mode)
        out = out.view(B, T, C, H, W)
        return out

class MCM(nn.Module):
    """Main MCM network for motion estimation."""
    def __init__(
        self,
        in_chans=2,
        out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[64, 128, 256, 512],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=2,
        inshape=(128, 128),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.spatial_dims = spatial_dims
        # Encoder
        self.encoder = MambaEncoder(in_chans, depths=depths, dims=feat_size, drop_path_rate=0.1)
        # Decoder
        self.decoder = DecoderWithDFH(encoder_channels=feat_size,
                                        decoder_channels=[256, 128, 64],
                                        nf=5,
                                        final_channels=out_chans)
        # Spatial transformer for registration
        self.spatial_trans = SpatialTransformer(inshape)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: source image sequence
            x2: target image sequence
        Returns:
            Deformed source and estimated deformation field
        """
        x = torch.cat((x1, x2), dim=2) 
        bz, nf, nc, h, w = x.shape
        encoder_out = self.encoder(x)
        deformation_field = self.decoder(encoder_out, target_h=h, target_w=w, bz=bz, nf=nf)
        x1_middle_frame = x1[:, nf // 2:(nf // 2 + 1), ...]
        deformed_x1 = self.spatial_trans(x1_middle_frame, deformation_field, mode="bilinear")
        return deformed_x1, deformation_field
