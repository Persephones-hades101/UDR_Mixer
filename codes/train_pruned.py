import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.dataset_utils import TrainDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR

from pruned_model import PrunedUDRMixer

class PrunedModel(pl.LightningModule):
    def __init__(self, config, fine_tune=True, original_model_path=None):
        super().__init__()
        self.net = PrunedUDRMixer(config)
        self.loss_fn = nn.L1Loss()
        
        # Optional - load weights from original model when fine-tuning
        if fine_tune and original_model_path:
            self._init_from_pretrained(original_model_path)
            
    def _init_from_pretrained(self, model_path):
        """Initialize weights from the original pruned model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        original_state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Transfer weights where shapes match
        model_state_dict = self.net.state_dict()
        for name, param in original_state_dict.items():
            if name in model_state_dict and param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name in model_state_dict:
                print(f"Shape mismatch for {name}: original {param.shape}, new {model_state_dict[name].shape}")
        
        print("Initialized weights from pretrained model")
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        degrad_patch, clean_patch = batch
        restored = self.net(degrad_patch)
        
        # Standard L1 loss
        loss = self.loss_fn(restored, clean_patch)
        
        # Add L1 regularization for filter pruning (soft pruning during fine-tuning)
        l1_reg = 0.0
        for param in self.net.parameters():
            l1_reg += torch.norm(param, 1)
        
        loss += 1e-5 * l1_reg  # Adjust weight as needed
        
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)  # Lower learning rate for fine-tuning
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, 
            warmup_epochs=5, 
            max_epochs=50
        )
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruned_model', type=str, default="pruned_model.pth")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='test_data_folder/')
    parser.add_argument('--ckpt_dir', type=str, default="pruned_checkpoints/")
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    
    # Load pruning configuration from the pruned model
    try:
        checkpoint = torch.load(args.pruned_model, map_location='cpu')
    except FileNotFoundError:
        print(f"ERROR: Pruned model file {args.pruned_model} not found!")
        return
    # Create pruning configuration based on masks
    mask_dict = checkpoint.get('mask_dict', {})
    pruned_config = {
        'base_dim': 64,
        'upscaling_factor': 2,
        'ffn_scale': 2.0,
    }
    
    # Add dimensions based on pruning masks (simplified)
    # In a real implementation, you would analyze the mask_dict
    # to determine actual dimensions after pruning
    if mask_dict:
        # Analyze mask_dict to determine actual dimensions after pruning
        print("Analyzing pruned model dimensions...")
        for layer_name, mask in mask_dict.items():
            remaining_filters = torch.sum(mask).item()
            total_filters = mask.numel()
            print(f"Layer {layer_name}: {remaining_filters}/{total_filters} filters remaining")
            
            # Set specific dimensions based on key layers
            if "to_feat1" in layer_name:
                pruned_config['feat1_out'] = int(remaining_filters)
            elif "feats1" in layer_name or "feats2" in layer_name:
                pruned_config['feat2_out'] = int(remaining_filters) 
            # Add more mappings as needed
    else:
        print("No mask_dict found in checkpoint, using default dimensions")
        pruned_config['feat1_out'] = 16  # Default example
        pruned_config['feat2_out'] = 64  # Default example
    
    print(f"Using pruned config: {pruned_config}")
    
    # Create dataset and dataloader
    opt = argparse.Namespace(**vars(args))
    try:
        trainset = TrainDataset(opt)
        print(f"Created dataset with {len(trainset)} samples")
        
        if len(trainset) == 0:
            print("ERROR: Dataset is empty! Check your data directory.")
            return
            
        trainloader = DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            pin_memory=True, 
            shuffle=True,
            drop_last=True, 
            num_workers=args.num_workers
        )
    except Exception as e:
        print(f"ERROR creating dataset/dataloader: {e}")
        import traceback
        traceback.print_exc()
        return
    # Create the pruned model
    model = PrunedModel(
        pruned_config,
        fine_tune=True,
        original_model_path=args.pruned_model
    )
    
    # Setup training
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        every_n_epochs=1,
        save_top_k=-1
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        callbacks=[checkpoint_callback]
    )
    
    # Train the pruned model
    trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    main()