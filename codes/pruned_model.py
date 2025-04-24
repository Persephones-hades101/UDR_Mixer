import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary components from the original model
from net.model import LayerNorm, FC, FFML, SFRL

class PrunedSFMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim)
        self.gobal = SFRL(dim)
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)
        y = self.gobal(y)
        y = self.fc(self.norm2(y)) + y
        return y

class PrunedFFMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.fft = FFML(dim)
        self.fc = FC(dim, ffn_scale)

    def forward(self, x):
        y = self.norm1(x)
        y = self.fft(y)
        y = self.fc(self.norm2(y)) + y
        return y

class PrunedDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PrunedDownsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class PrunedUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PrunedUpsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class PrunedUDRMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Configuration contains pruned dimensions
        self.config = config
        
        # Extract dimensions from config
        dim = config.get('base_dim', 64)
        upscaling_factor = config.get('upscaling_factor', 2)
        ffn_scale = config.get('ffn_scale', 2.0)
        
        # Dimensions may vary based on pruning
        feat1_out = config.get('feat1_out', dim // 4)
        feat2_out = config.get('feat2_out', dim)
        
        # Initial feature extraction
        self.to_feat1 = nn.Conv2d(3, feat1_out, 3, 1, 1)
        self.to_feat2 = nn.PixelUnshuffle(upscaling_factor)
        
        # SFMB blocks with potentially pruned dimensions
        self.feats1 = PrunedSFMB(feat2_out, ffn_scale)
        self.feats2 = PrunedSFMB(feat2_out, ffn_scale)
        
        # Downsample with specific in/out channels
        down1_2_out = config.get('down1_2_out', dim*2)
        self.down1_2 = PrunedDownsample(feat2_out, down1_2_out // 4)
        
        # Middle level features
        mid_dim = config.get('mid_dim', dim*2)
        self.feats3 = PrunedSFMB(mid_dim, ffn_scale)
        self.feats4 = PrunedSFMB(mid_dim, ffn_scale)
        
        # FFT branch dimensions
        fft_dim = config.get('fft_dim', 32)
        fft_out = config.get('fft_cat_dim', 2048)
        
        # Dimensions after concatenation
        cat_dim = config.get('cat_dim', mid_dim + fft_out)
        reduce_dim = config.get('reduce_dim', mid_dim)
        
        # Remaining blocks
        self.feats6 = PrunedSFMB(reduce_dim, ffn_scale)
        self.feats7 = PrunedSFMB(reduce_dim, ffn_scale)
        
        # Upsample with specific in/out channels
        # ─── Upsample (same) ──────────────────────────────────────────────
        up2_1_in  = config.get('up2_1_in',  reduce_dim)
        up2_1_out = config.get('up2_1_out', feat2_out)          # 44 in your run
        self.up2_1 = PrunedUpsample(up2_1_in, up2_1_out * 2)    # PixelShuffle(2)

        # ─── Compute true concat width after PixelShuffle ────────────────
        # PixelShuffle(2) divides channels by 4
        upsampled_ch = (up2_1_out * 2) // 4                     # e.g. (44*2)//4 = 22
        cat2_dim     = feat2_out + upsampled_ch                 # skip(44) + up(22) = 66

        # conv that follows the concatenation
        self.reduce_chan_level2 = nn.Conv2d(
            in_channels=cat2_dim,
            out_channels=feat2_out,
            kernel_size=1,
            bias=False
        )
                
        self.feats8 = PrunedSFMB(feat2_out, ffn_scale)
        self.feats9 = PrunedSFMB(feat2_out, ffn_scale)
        
        # Output layers
        self.to_img1 = nn.Conv2d(feat2_out, 48, 3, 1, 1)
        self.to_img2 = nn.PixelShuffle(4)
        
        # FFT branch
        self.to_feat_fft = nn.Sequential(
            nn.Conv2d(3, fft_dim, 3, 1, 1)
        )
        self.feats_fft = nn.Sequential(*[PrunedFFMB(fft_dim, ffn_scale) for _ in range(4)])
        self.down_fft = nn.PixelUnshuffle(8)
        self.reduce_chan_fft = nn.Conv2d(cat_dim, reduce_dim, 3, 1, 1)
    
    def forward(self, x):
        # FFT branch
        x_fft = x
        x_fft = self.to_feat_fft(x_fft)
        x_fft = self.feats_fft(x_fft)
        x_fft = self.down_fft(x_fft)
        
        # Main branch
        x = F.interpolate(x, scale_factor=1/2, mode='bicubic', align_corners=False)
        x = self.to_feat1(x)
        x = self.to_feat2(x)
        x1 = x
        x = self.feats1(x)
        x = self.feats2(x)
        x_skip = x
        
        x = self.down1_2(x)
        x = self.feats3(x)
        x = self.feats4(x)
        
        # Concatenate with FFT branch
        x = torch.cat([x, x_fft], dim=1)
        x = self.reduce_chan_fft(x)
        
        x = self.feats6(x)
        x = self.feats7(x)
        x = self.up2_1(x)
        
        # Skip connection
        x = torch.cat([x, x_skip], 1)
        x = self.reduce_chan_level2(x)
        
        x = self.feats8(x)
        x = self.feats9(x)
        x = self.to_img1(x + x1)
        x = self.to_img2(x)
        
        return x