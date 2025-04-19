import sys
import torch
from net.model import UDRMixer
from torchinfo import summary

def model_summary():
    # Create a model instance
    model = UDRMixer(dim=64, n_blocks=8, ffn_scale=2.0, upscaling_factor=2)
    
    # Create a sample input tensor (batch_size, channels, height, width)
    input_size = (1, 3, 512, 512)  # Adjust height/width as needed for your use case
    
    # Generate and print the summary
    model_stats = summary(model, input_size=input_size, depth=3, 
                          col_names=["input_size", "output_size", "num_params", "trainable"])
    
    print(model_stats)

if __name__ == "__main__":
    model_summary()