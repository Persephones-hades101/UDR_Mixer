import argparse
import torch
from net.model import UDRMixer
from pruning.filter_pruning import l1_norm_pruning, apply_filter_pruning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="model/model.ckpt", help='checkpoint save path')
    parser.add_argument('--prune_ratio', type=float, default=0.3, help='pruning ratio (0-1)')
    parser.add_argument('--output_path', type=str, default="model/pruned_model.ckpt", help='output model path')
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    
    # Create a new model
    model = UDRMixer(dim=64, n_blocks=8, ffn_scale=2.0, upscaling_factor=2)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Exclude certain layers from pruning (optional)
    exclude_layers = [
        'net.head',  # First layer
        'net.tail'   # Last layer
    ]
    
    # Analyze model for pruning
    pruned_model, pruned_cfg = l1_norm_pruning(model, args.prune_ratio, exclude_layers)
    
    # Apply pruning to create new model
    new_model = apply_filter_pruning(model, pruned_cfg)
    
    # Save the pruned model
    torch.save({
        'state_dict': new_model.state_dict(),
        'pruned_cfg': pruned_cfg,
        'original_model': 'UDRMixer',
        'prune_ratio': args.prune_ratio
    }, args.output_path)
    
    print(f"Pruned model saved to {args.output_path}")
    
    # Print model size comparison
    original_size = sum(p.numel() for p in model.parameters())
    pruned_size = sum(p.numel() for p in new_model.parameters())
    compression_ratio = (original_size - pruned_size) / original_size * 100
    
    print(f"Original model parameters: {original_size:,}")
    print(f"Pruned model parameters: {pruned_size:,}")
    print(f"Compression ratio: {compression_ratio:.2f}%")

if __name__ == '__main__':
    main()