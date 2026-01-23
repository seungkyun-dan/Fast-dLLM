import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_layer_data

def calculate_entropy(attn_map):
    """Computes mean entropy per head. Input: (Heads, Q, K)"""
    # Avoid log(0)
    attn_map = torch.clamp(attn_map, min=1e-9)
    # Entropy = -sum(p * log(p)) over K dim
    entropy = -torch.sum(attn_map * torch.log(attn_map), dim=-1)
    # Mean over Q dim -> Result: (Heads,)
    return entropy.mean(dim=-1)

def main(args):
    data = load_layer_data(args.dir, args.step, args.layer)
    if data is None:
        print("Data not found.")
        return

    entropies = calculate_entropy(data).numpy()
    heads = np.arange(len(entropies))
    
    # Identify sharpest head
    min_idx = np.argmin(entropies)
    min_val = entropies[min_idx]

    # Plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(heads, entropies, color='skyblue')
    bars[min_idx].set_color('red') # Highlight sharpest
    
    plt.axhline(y=entropies.mean(), color='gray', linestyle='--')
    plt.title(f"Head Entropy (Step {args.step}, Layer {args.layer})\nSharpest: Head {min_idx} (H={min_val:.3f})")
    plt.xlabel("Head Index")
    plt.ylabel("Entropy")
    
    out_path = f"entropy_s{args.step}_l{args.layer}.png"
    plt.savefig(out_path)
    print(f"Saved: {out_path} (Sharpest Head: {min_idx})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to attn_maps directory")
    parser.add_argument("--step", type=int, default=10, help="Target step")
    parser.add_argument("--layer", type=int, default=15, help="Target layer")
    args = parser.parse_args()
    main(args)