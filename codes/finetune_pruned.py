import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.dataset_utils import TrainDataset
from net.pruned_model import PrunedUDRMixer
from utils.schedulers import LinearWarmupCosineAnnealingLR

class PrunedModel(pl.LightningModule):
    def __init__(self, pruned_model):
        super().__init__()
        self.net = pruned_model
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)  # Lower learning rate for fine-tuning
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5, max_epochs=50)
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruned_ckpt', type=str, default="model/pruned_model.ckpt", help='pruned checkpoint path')
    parser.add_argument('--data_dir', type=str, default="dataset/", help='path to training data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--patch_size', type=int, default=128, help='patch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--ckpt_dir', type=str, default="model/pruned_finetuned/", help='finetuned model save path')
    args = parser.parse_args()
    
    # Load the pruned model
    print(f"Loading pruned model from {args.pruned_ckpt}")
    checkpoint = torch.load(args.pruned_ckpt, map_location='cpu')
    pruned_cfg = checkpoint['pruned_cfg']
    
    # Create the pruned model
    pruned_model = PrunedUDRMixer(dim=64, n_blocks=8, ffn_scale=2.0, upscaling_factor=2, cfg=pruned_cfg)
    pruned_model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Create dataset and dataloader
    train_dataset = TrainDataset(args)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create the Lightning model and trainer
    model = PrunedModel(pruned_model)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        every_n_epochs=5,
        save_top_k=-1
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        callbacks=[checkpoint_callback]
    )
    
    # Fine-tune the model
    trainer.fit(model=model, train_dataloaders=train_loader)
    
    # Save the final model
    torch.save({
        'state_dict': model.state_dict(),
        'pruned_cfg': pruned_cfg,
        'prune_ratio': checkpoint['prune_ratio']
    }, f"{args.ckpt_dir}/final_model.ckpt")
    
    print(f"Fine-tuned model saved to {args.ckpt_dir}/final_model.ckpt")

if __name__ == '__main__':
    main()