import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import torch
from tqdm import tqdm

def load_data_safely(file_path):
    """
    Load PyTorch tensor from file.
    map_location='cpu' ensures it works on Mac/CPU environments
    even if the data was saved from a CUDA GPU.
    """
    try:
        data = torch.load(file_path, map_location='cpu')
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_layer_heads(data, step, layer, output_dir):
    """
    Visualize attention maps for all heads in a single layer.
    Saves the result as a single PNG file.
    """
    # Automatically detect number of heads
    num_heads = data.shape[0]
    
    # Configure grid (8 columns fixed)
    ncols = 8
    nrows = math.ceil(num_heads / ncols)
    
    # Setup canvas size
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 3 * nrows))
    axes = axes.flatten()
    
    # Plot each head
    for h in range(num_heads):
        ax = axes[h]
        # Convert tensor to float and then numpy for plotting
        attn_map = data[h].float().numpy()
        
        # Draw heatmap with colorbar
        sns.heatmap(
            attn_map, 
            cmap="viridis",   # 'viridis' colormap similar to the reference
            vmin=0.0,         # Fix min value to 0
            vmax=1.0,         # Fix max value to 1
            square=True, 
            cbar=True, 
            ax=ax, 
            xticklabels=False, 
            yticklabels=False,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"Head {h}", fontsize=10)
    
    # Remove empty subplots
    for h in range(num_heads, len(axes)):
        fig.delaxes(axes[h])
        
    plt.suptitle(f"LLaDA 8B Attention - Step {step} | Layer {layer}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save file: step_XXX_layer_XX.png (Flat structure)
    save_name = f"step_{step:03d}_layer_{layer:02d}.png"
    save_path = os.path.join(output_dir, save_name)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # Close figure to free memory

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_dir = args.dir
    print(f"Source Directory: {base_dir}")
    print(f"Output Directory: {args.output_dir}")

    # 1. Find and sort step directories
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return

    step_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("step_") and os.path.isdir(os.path.join(base_dir, d))])
    
    if not step_dirs:
        print("No 'step_XXX' directories found.")
        return

    print(f"Found {len(step_dirs)} step directories. Starting processing...")

    # 2. Iterate through step directories
    for step_dir in tqdm(step_dirs, desc="Total Progress"):
        try:
            # Extract step number from folder name
            step_num = int(step_dir.split('_')[1])
        except ValueError:
            continue

        step_path = os.path.join(base_dir, step_dir)
        
        # 3. Find layer files in the step directory
        layer_files = sorted([f for f in os.listdir(step_path) if f.startswith("layer_") and f.endswith(".pt")])
        
        if not layer_files:
            continue

        # 4. Process each layer
        for l_file in tqdm(layer_files, desc=f"Processing {step_dir}", leave=False):
            try:
                # Extract layer number from filename
                layer_str = l_file.split('_')[1].split('.')[0]
                layer_num = int(layer_str)
            except (IndexError, ValueError):
                continue

            file_path = os.path.join(step_path, l_file)
            
            # Load data
            data = load_data_safely(file_path)
            if data is None:
                continue

            # Plot and save
            plot_layer_heads(data, step_num, layer_num, args.output_dir)

    print(f"\nAll Done! Images saved to '{args.output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Input path
    parser.add_argument("--dir", type=str, required=True, help="Path to input directory containing step_XXX folders")
    
    # Output path
    parser.add_argument("--output_dir", type=str, default="./attn_maps_sar", help="Directory to save output images")
    
    args = parser.parse_args()
    main(args)