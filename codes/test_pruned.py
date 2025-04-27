# soft_pruned_test.py
import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import lightning.pytorch as pl
import torch.nn.functional as F

from utils.dataset_utils import TestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from pruned_model import PrunedUDRMixer   # <--- IMPORTANT: your pruned model

class SoftPrunedModel(pl.LightningModule):
    def __init__(self, pruned_config):
        super().__init__()
        self.net = PrunedUDRMixer(pruned_config)
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        return optimizer

def test(net, dataset, dataset_name=""):
    output_path = testopt.output_path + dataset_name + '/'
    os.makedirs(output_path, exist_ok=True)
    
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for (degraded_name, degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--data_path', type=str, default="test_data_folder/", help='path to test degraded images')
    parser.add_argument('--output_path', type=str, default="output/", help='path to save output images')
    parser.add_argument('--ckpt_path', type=str, default="pruned_checkpoints/last.ckpt", help='path to model checkpoint')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    # Load checkpoint
    ckpt = torch.load(testopt.ckpt_path, map_location="cpu")

    # Prepare pruning config
    mask_dict = ckpt.get('mask_dict', {})
    pruned_config = {
        'base_dim': 64,
        'upscaling_factor': 2,
        'ffn_scale': 2.0,
    }

    if mask_dict:
        for layer_name, mask in mask_dict.items():
            if "to_feat1" in layer_name:
                remaining_filters = int(torch.sum(mask).item())
                pruned_config['feat1_out'] = remaining_filters
    # after loop
    feat1 = pruned_config.get('feat1_out', 16)
    pruned_config['feat2_out'] = feat1 * 4  # pixel unshuffle effect

print(f"Using pruned config: {pruned_config}")

    # Build model
    net = SoftPrunedModel(pruned_config)
    ckpt = torch.load(testopt.ckpt_path, map_location="cpu")

    # Load matching weights
    model_state = net.state_dict()
    for name, param in ckpt['state_dict'].items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name].copy_(param)

    net.eval()
    net = net.cuda()

    # Load test dataset
    data_set = TestDataset(testopt)

    # Run testing
    test(net, data_set)
