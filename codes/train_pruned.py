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
    parser.add_argument('--batch_size', type=int, default=1)  # Reduce batch size to 1
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='test_data_folder/')
    parser.add_argument('--ckpt_dir', type=str, default="pruned_checkpoints/")
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)  # Set to 0 to avoid worker issues
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Check data availability and structure
    input_dir = os.path.join(args.data_dir, "input")
    target_dir = os.path.join(args.data_dir, "target")
    
    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        print(f"ERROR: Input or target directory missing. Check {input_dir} and {target_dir}")
        return
    
    input_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    target_files = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(input_files)} input files and {len(target_files)} target files")
    
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
    
    # Add dimensions based on pruning masks
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
    else:
        print("No mask_dict found in checkpoint, using default dimensions")
        pruned_config['feat1_out'] = 16  # Default example
        pruned_config['feat2_out'] = 64  # Default example
    
    print(f"Using pruned config: {pruned_config}")
    
    # Create custom namespace for dataset
    opt = argparse.Namespace(**vars(args))
    
    # Debug: Examine a specific image to ensure it can be loaded properly
    if args.debug:
        from PIL import Image
        try:
            sample_input = os.path.join(input_dir, input_files[0])
            sample_target = os.path.join(target_dir, target_files[0])
            print(f"Testing image loading with: {sample_input}")
            img = Image.open(sample_input)
            print(f"Successfully loaded input image with size: {img.size}")
            img = Image.open(sample_target)
            print(f"Successfully loaded target image with size: {img.size}")
        except Exception as e:
            print(f"Error loading test image: {e}")
            return
    
    # Create dataset with explicit error handling
    try:
        trainset = TrainDataset(opt)
        print(f"Created dataset with {len(trainset)} samples")
        
        if len(trainset) == 0:
            print("ERROR: Dataset is empty! Check your data directory.")
            return
        
        # Debug: Test fetching an item from dataset
        if args.debug:
            try:
                print("Testing dataset __getitem__")
                sample = trainset[0]
                if isinstance(sample, tuple) and len(sample) == 2:
                    print(f"Successfully fetched sample: {[s.shape for s in sample]}")
                else:
                    print(f"Unexpected sample format: {type(sample)}")
            except Exception as e:
                print(f"Error fetching sample from dataset: {e}")
                import traceback
                traceback.print_exc()
                return
            
        # Create dataloader with proper error handling
        trainloader = DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            pin_memory=True, 
            shuffle=True,
            drop_last=False,  # Change to False to ensure small batches are used
            num_workers=args.num_workers
        )
        
        # Debug: Test dataloader
        if args.debug:
            print("Testing DataLoader iteration")
            try:
                for i, batch in enumerate(trainloader):
                    print(f"Successfully loaded batch {i}: {[b.shape for b in batch]}")
                    if i >= 2:  # Just check a few batches
                        break
            except Exception as e:
                print(f"Error iterating through DataLoader: {e}")
                import traceback
                traceback.print_exc()
                return
        
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
    
    # Create trainer with specific logger
    from lightning.pytorch.loggers import CSVLogger
    logger = CSVLogger(save_dir=args.ckpt_dir, name="pruned_training_logs")
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
    )
    
    # Train the pruned model with additional error handling
    try:
        print("Starting model training...")
        trainer.fit(model=model, train_dataloaders=trainloader)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()