import torch
import torch.nn as nn
import numpy as np
from net.model import UDRMixer
import copy

def calculate_filter_importance(model):
    """Calculate importance of filters based on L1-norm."""
    importance_dict = {}
    
    # Iterate through all Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate L1 norm for each filter
            weights = module.weight.data
            importance = torch.sum(torch.abs(weights), dim=[1, 2, 3])
            importance_dict[name] = importance
            
    return importance_dict

def create_pruning_mask(importance_dict, prune_ratio=0.3):
    """Create binary masks for each layer based on importance."""
    mask_dict = {}
    
    for name, importance in importance_dict.items():
        n_filters = len(importance)
        n_to_keep = int(n_filters * (1 - prune_ratio))
        
        # Sort filters by importance
        _, indices = torch.sort(importance, descending=True)
        keep_indices = indices[:n_to_keep]
        
        # Create binary mask (1 for keep, 0 for prune)
        mask = torch.zeros_like(importance)
        mask[keep_indices] = 1
        mask_dict[name] = mask
        
    return mask_dict

def apply_masks_to_model(model, mask_dict):
    """Apply pruning masks to model (soft pruning)."""
    for name, module in model.named_modules():
        if name in mask_dict and isinstance(module, nn.Conv2d):
            mask = mask_dict[name]
            # Apply mask to filters
            module.weight.data = module.weight.data * mask.view(-1, 1, 1, 1)
            
            # If bias exists, also prune corresponding bias
            if module.bias is not None:
                module.bias.data = module.bias.data * mask
    
    return model

def analyze_model_structure(model):
    """Analyze model structure to handle dependencies between layers."""
    # Map from layer output to layers that use it as input
    dependency_map = {}
    
    # Track layer dimensions
    layer_dims = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_dims[name] = {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels
            }
    
    # Define key connections that need to be synchronized
    key_connections = [
        ('feats2', 'down1_2'),
        ('down1_2', 'feats3'),
        ('feats4', 'reduce_chan_fft'),
        ('feats7', 'up2_1'),
        ('up2_1', 'reduce_chan_level2')
    ]
    
    for source, target in key_connections:
        if source not in dependency_map:
            dependency_map[source] = []
        dependency_map[source].append(target)
    
    return dependency_map, layer_dims

def propagate_pruning(mask_dict, dependency_map):
    """Propagate pruning decisions to dependent layers."""
    for source, targets in dependency_map.items():
        if source in mask_dict:
            source_mask = mask_dict[source]
            # For each dependent layer, ensure consistency
            for target in targets:
                if target in mask_dict:
                    # The output channels of source must match input channels of target
                    # For UDR-Mixer, this requires careful handling of channel dimensions
                    # (simplified for this example)
                    pass
    
    return mask_dict

def evaluate_pruned_model(model, test_loader):
    """Evaluate pruned model to check performance."""
    model.eval()
    avg_psnr = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            degrad_patch, clean_patch = batch
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            
            output = model(degrad_patch)
            # Calculate PSNR (implementation depends on your metrics)
            # psnr = compute_psnr(output, clean_patch)
            # avg_psnr += psnr
            count += 1
    
    return avg_psnr / count if count > 0 else 0

def prune_udr_mixer(model_path, prune_ratio=0.3, save_path="pruned_model.pth"):
    """Main pruning function."""
    # Load the model
    model = UDRMixer(dim=64, n_blocks=8, ffn_scale=2.0, upscaling_factor=2)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()
    
    # Calculate filter importance
    importance_dict = calculate_filter_importance(model)
    
    # Create pruning masks
    mask_dict = create_pruning_mask(importance_dict, prune_ratio)
    
    # Analyze model structure and dependencies
    dependency_map, layer_dims = analyze_model_structure(model)
    
    # Ensure consistent pruning across dependent layers
    mask_dict = propagate_pruning(mask_dict, dependency_map)
    
    # Apply masks to the model (soft pruning)
    pruned_model = apply_masks_to_model(copy.deepcopy(model), mask_dict)
    
    # Save the pruned model
    torch.save({
        'state_dict': pruned_model.state_dict(),
        'prune_ratio': prune_ratio,
        'mask_dict': mask_dict
    }, save_path)
    
    print(f"Model pruned with ratio {prune_ratio} and saved to {save_path}")
    return pruned_model, mask_dict

def check_data_availability(data_dir):
    """Check if training data is available and properly structured."""
    import os
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")
    
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory {input_dir} does not exist!")
        return False
    
    if not os.path.exists(target_dir):
        print(f"ERROR: Target directory {target_dir} does not exist!")
        return False
    
    input_files = [f for f in os.listdir(input_dir) if not f.startswith('.')]
    target_files = [f for f in os.listdir(target_dir) if not f.startswith('.')]
    
    print(f"Found {len(input_files)} input files and {len(target_files)} target files")
    
    if len(input_files) == 0 or len(target_files) == 0:
        print("ERROR: No training data found!")
        return False
    
    # Verify matching files
    input_basenames = set([os.path.splitext(f)[0] for f in input_files])
    target_basenames = set([os.path.splitext(f)[0] for f in target_files])
    common_files = input_basenames.intersection(target_basenames)
    
    print(f"Found {len(common_files)} matching input/target pairs")
    
    if len(common_files) == 0:
        print("ERROR: No matching input/target pairs found!")
        return False
    
    return True

if __name__ == "__main__":
    model_path = "model/model.ckpt"  # Path to your trained model
    pruned_model, mask_dict = prune_udr_mixer(model_path, prune_ratio=0.3)