import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

def load_data_safely(file_path):
    """
    Load PyTorch tensor from file.
    map_location='cpu' ensures it works on Mac/CPU environments
    """
    try:
        data = torch.load(file_path, map_location='cpu')
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_layer_step_diff(curr_data, prev_data, layer, step, output_dir):
    """
    Visualize the Absolute Difference between Current Step and Previous Step for each head.
    Subplot titles include the Cosine Similarity between the two steps' heads.
    """
    num_heads = curr_data.shape[0]
    
    # Configure grid (8 columns fixed)
    ncols = 8
    nrows = math.ceil(num_heads / ncols)
    
    # Setup canvas size
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 3 * nrows))
    axes = axes.flatten()
    
    # Plot each head
    for h in range(num_heads):
        ax = axes[h]
        
        # Get tensors for this head
        curr_head = curr_data[h].float()
        prev_head = prev_data[h].float()
        
        # Calculate Cosine Similarity (Scalar)
        # Flatten to vectors
        cos_sim = F.cosine_similarity(curr_head.flatten(), prev_head.flatten(), dim=0, eps=1e-8).item()
        
        # Calculate Difference Map (Per Element Comparison)
        # Using Absolute Difference to see magnitude of change
        diff_map = torch.abs(curr_head - prev_head).numpy()
        
        # Draw heatmap
        sns.heatmap(
            diff_map, 
            cmap="viridis",
            vmin=0.0,
            vmax=0.5, # Reduced vmax since differences might be small. Adjusted for visibility.
            square=True, 
            cbar=True, 
            ax=ax, 
            xticklabels=False, 
            yticklabels=False,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"Head {h}\nSim: {cos_sim:.4f}", fontsize=10)
    
    # Remove empty subplots
    for h in range(num_heads, len(axes)):
        fig.delaxes(axes[h])
        
    plt.suptitle(f"LLaDA 8B Step {step} vs {step-1} Diff - Layer {layer}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save file: step_XXX_layer_XX.png
    save_name = f"step_{step:03d}_layer_{layer:02d}.png"
    save_path = os.path.join(output_dir, save_name)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_dir = args.dir
    print(f"Source Directory: {base_dir}")
    print(f"Output Directory: {args.output_dir}")

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return

    # 1. Find and sort step directories
    step_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("step_") and os.path.isdir(os.path.join(base_dir, d))])
    
    if not step_dirs:
        print("No 'step_XXX' directories found.")
        return

    # Extract step numbers info
    steps_info = []
    for d in step_dirs:
        try:
            num = int(d.split('_')[1])
            steps_info.append((num, d))
        except ValueError:
            continue
    
    # Sort by step number
    steps_info.sort(key=lambda x: x[0])
    sorted_steps = steps_info # List of (step_num, step_dir_name)

    print(f"Found {len(sorted_steps)} steps. (From {sorted_steps[0][0]} to {sorted_steps[-1][0]})")

    # 2. Identify available layers
    first_step_path = os.path.join(base_dir, sorted_steps[0][1])
    layer_files = sorted([f for f in os.listdir(first_step_path) if f.startswith("layer_") and f.endswith(".pt")])
    
    layers = []
    for lf in layer_files:
        try:
            l_num = int(lf.split('_')[1].split('.')[0])
            layers.append(l_num)
        except:
            continue
    layers.sort()
    
    print(f"Found {len(layers)} layers. Starting processing...")
    
    # Cache for previous step data: { layer_num: Tensor }
    prev_step_data = {} 

    # Processing Loop
    for i, (step_num, step_dir) in enumerate(tqdm(sorted_steps, desc="Processing Steps")):
        step_path = os.path.join(base_dir, step_dir)
        
        # If it's the very first step in the list, we just cache it and continue?
        # Or do we want to output something? user said "compare step 0 and 1".
        # So for Step 0, there is no Step -1. We can skip plotting or plot raw.
        # Let's skip plotting for the first step, just cache.
        
        for layer_num in layers:
            file_name = f"layer_{layer_num:02d}.pt"
            file_path = os.path.join(step_path, file_name)
            
            if not os.path.exists(file_path):
                if layer_num in prev_step_data: del prev_step_data[layer_num]
                continue
                
            curr_data = load_data_safely(file_path)
            if curr_data is None:
                if layer_num in prev_step_data: del prev_step_data[layer_num]
                continue
            
            # Check if we have previous data to compare with
            if layer_num in prev_step_data:
                prev_data = prev_step_data[layer_num]
                
                # Check shapes
                if prev_data.shape == curr_data.shape:
                    plot_layer_step_diff(curr_data, prev_data, layer_num, step_num, args.output_dir)
            
            # Update cache
            prev_step_data[layer_num] = curr_data

    print(f"\nAll Done! Images saved to '{args.output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Input path
    parser.add_argument("--dir", type=str, required=True, help="Path to input directory containing step_XXX folders")
    
    # Output path
    parser.add_argument("--output_dir", type=str, default="./attn_maps_sar/step_sim", help="Directory to save output images")
    
    args = parser.parse_args()
    main(args)
