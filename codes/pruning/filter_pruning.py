import torch
import torch.nn as nn
import numpy as np
import copy

def l1_norm_pruning(model, prune_ratio, exclude_layers=None):
    """
    Prune filters with the lowest L1 norm from each convolutional layer.
    
    Args:
        model: The neural network model
        prune_ratio: Percentage of filters to prune in each layer
        exclude_layers: List of layer names to exclude from pruning
    
    Returns:
        pruned_model: Model with pruned filters
        cfg: Configuration showing the new number of filters in each layer
    """
    if exclude_layers is None:
        exclude_layers = []
    
    # Make a copy of the model to avoid modifying the original
    pruned_model = copy.deepcopy(model)
    
    # Store the original filter counts and the new pruned counts
    original_cfg = {}
    pruned_cfg = {}
    
    # Collect all convolutional layers
    conv_layers = []
    layer_names = []
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) and name not in exclude_layers:
            # Only include layers with out_channels > 1 (avoid pruning depthwise convs)
            if module.out_channels > 1:
                conv_layers.append(module)
                layer_names.append(name)
                original_cfg[name] = module.out_channels
    
    for i, (name, layer) in enumerate(zip(layer_names, conv_layers)):
        # Calculate L1-norm for each filter
        weight = layer.weight.data.clone()
        num_filters = weight.size(0)
        l1_norm = torch.sum(torch.abs(weight.view(num_filters, -1)), dim=1)
        
        # Calculate number of filters to keep
        num_to_keep = int(num_filters * (1 - prune_ratio))
        num_to_keep = max(1, num_to_keep)  # Keep at least one filter
        
        # Find indices of filters to keep
        _, indices = torch.topk(l1_norm, num_to_keep)
        mask = torch.zeros(num_filters)
        mask[indices] = 1
        
        # Record the pruned configuration
        pruned_cfg[name] = num_to_keep
        
        print(f"Pruning {name}: {num_filters} -> {num_to_keep} filters")
        
    return pruned_model, pruned_cfg

def apply_filter_pruning(model, pruned_cfg):
    """
    Create a new pruned model according to the pruned configuration.
    
    Args:
        model: Original model
        pruned_cfg: Dictionary of layer name to number of filters to keep
    
    Returns:
        new_model: The pruned model
    """
    # Create a new model with the same architecture
    new_model = copy.deepcopy(model)
    
    # Dictionary to store pruned weights for each layer
    pruned_weights = {}
    
    # First pass: Extract indices of filters to keep for each layer
    for name, module in new_model.named_modules():
        if name in pruned_cfg and isinstance(module, nn.Conv2d):
            # Calculate L1-norm for each filter
            weight = module.weight.data.clone()
            num_filters = weight.size(0)
            l1_norm = torch.sum(torch.abs(weight.view(num_filters, -1)), dim=1)
            
            # Get indices of filters to keep
            num_to_keep = pruned_cfg[name]
            _, indices = torch.topk(l1_norm, num_to_keep)
            indices, _ = torch.sort(indices)  # Sort indices
            
            # Store indices for this layer
            pruned_weights[name] = {'indices': indices}
    
    # Second pass: Rebuild the model with pruned filters
    for name, module in new_model.named_modules():
        if name in pruned_weights and isinstance(module, nn.Conv2d):
            # Get indices of filters to keep for this layer
            indices = pruned_weights[name]['indices']
            
            # Extract weights for kept filters
            new_weights = module.weight.data[indices]
            
            # Create new conv layer with pruned filters
            new_conv = nn.Conv2d(
                in_channels=module.in_channels, 
                out_channels=len(indices),
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None)
            )
            
            # Set the pruned weights
            new_conv.weight.data = new_weights
            if module.bias is not None:
                new_conv.bias.data = module.bias.data[indices]
            
            # Replace the original module with the pruned one
            # Note: This is a simplified approach and may not work for complex architectures
            # For complex models, you would need to rebuild the entire model
            setattr(module.__class__, name, new_conv)
    
    return new_model