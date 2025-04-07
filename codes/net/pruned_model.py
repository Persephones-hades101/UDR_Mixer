import torch
import torch.nn as nn
import torch.nn.functional as F
from net.model import LayerNorm, FC, FFML, SFRL

class PrunedFFMB(nn.Module):
    def __init__(self, dim, cfg, block_idx):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
        # Use pruned configuration for conv layers
        self.conv1 = nn.Conv2d(dim, cfg[f'net.blocks.{block_idx}.conv1'], 1, 1, 0)
        self.conv2 = nn.Conv2d(cfg[f'net.blocks.{block_idx}.conv1'], cfg[f'net.blocks.{block_idx}.conv2'], 3, 1, 1)
        self.conv3 = nn.Conv2d(cfg[f'net.blocks.{block_idx}.conv2'], dim, 1, 1, 0)
        
        # FFT branch
        self.fft = FFML(dim)
        
        # Spatial branch
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, cfg[f'net.blocks.{block_idx}.spatial.0.weight'], 3, 1, 1, groups=1),
            nn.GELU(),
            nn.Conv2d(cfg[f'net.blocks.{block_idx}.spatial.0.weight'], dim, 1, 1, 0)
        )
        
        # Global features
        self.global_block = SFRL(dim)

    def forward(self, x):
        # First pathway
        y = self.norm1(x)
        y = self.conv1(y)
        y = F.gelu(y)
        y = self.conv2(y)
        y = F.gelu(y)
        y = self.conv3(y)
        
        # Second pathway - FFT
        z = self.fft(x)
        
        # Third pathway - Spatial
        w = self.spatial(x)
        
        # Fourth pathway - Global
        g = self.global_block(x)
        
        # Combine all pathways
        out = y + z + w + g + x
        
        return out

class PrunedUDRMixer(nn.Module):
    def __init__(self, dim=64, n_blocks=8, ffn_scale=2.0, upscaling_factor=2, cfg=None):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        self.upscaling_factor = upscaling_factor
        
        # Default configuration if none provided
        if cfg is None:
            cfg = {}
            for i in range(n_blocks):
                cfg[f'net.blocks.{i}.conv1'] = dim
                cfg[f'net.blocks.{i}.conv2'] = dim
                cfg[f'net.blocks.{i}.spatial.0.weight'] = dim
        
        # Head
        self.head = nn.Conv2d(3, dim, 3, 1, 1)
        
        # Blocks
        self.blocks = nn.ModuleList([
            PrunedFFMB(dim, cfg, i) for i in range(n_blocks)
        ])
        
        # Tail
        if upscaling_factor == 2:
            self.tail = nn.Sequential(
                nn.Conv2d(dim, dim * ffn_scale, 3, 1, 1),
                nn.PixelShuffle(upscaling_factor),
                nn.Conv2d(dim // (upscaling_factor), 3, 3, 1, 1)
            )
        else:
            self.tail = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        # Extract features
        feats = self.head(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            feats = block(feats)
        
        # Reconstruction
        out = self.tail(feats)
        
        return out + F.interpolate(x, scale_factor=self.upscaling_factor, mode='bilinear', align_corners=False)