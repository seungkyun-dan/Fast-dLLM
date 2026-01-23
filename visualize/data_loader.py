import os
import glob
import torch

def get_sorted_steps(base_dir):
    """Returns a list of step numbers (integers) sorted descending (T -> 0)."""
    pattern = os.path.join(base_dir, "step_*")
    dirs = glob.glob(pattern)
    steps = []
    for d in dirs:
        try:
            # Extract number from "step_050"
            num = int(os.path.basename(d).split("_")[-1])
            steps.append(num)
        except ValueError:
            continue
    return sorted(steps, reverse=True)

def load_layer_data(base_dir, step, layer_idx):
    """Loads tensor for specific step and layer. Returns (Heads, Q, K)."""
    fname = f"layer_{layer_idx:02d}.pt"
    path = os.path.join(base_dir, f"step_{step:03d}", fname)
    
    if not os.path.exists(path):
        return None
        
    # Load to CPU and float32 for analysis
    try:
        data = torch.load(path, map_location="cpu").float()
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None