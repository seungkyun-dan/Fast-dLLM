import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_sorted_steps, load_layer_data

def main(args):
    steps = get_sorted_steps(args.dir)
    if not steps:
        print(f"No data found in {args.dir}")
        return

    print(f"Processing Layer {args.layer} over {len(steps)} steps...")
    
    sink_history = []
    valid_steps = []

    for step in steps:
        # Data shape: (Heads, Q, K)
        data = load_layer_data(args.dir, step, args.layer)
        if data is None: continue

        # 1. Mean over heads -> (Q, K)
        # 2. Sum over Query dim (Incoming Attention) -> (K,)
        incoming_attn = data.mean(dim=0).sum(dim=0)
        
        sink_history.append(incoming_attn.numpy())
        valid_steps.append(step)

    # Create 2D Map: (Steps, Seq_Len)
    sink_map = np.stack(sink_history)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.heatmap(sink_map, cmap="viridis", yticklabels=valid_steps)
    
    plt.title(f"Attention Sink Evolution (Layer {args.layer})")
    plt.xlabel("Token Index")
    plt.ylabel("Denoising Step")
    plt.tight_layout()
    
    out_path = f"sink_layer_{args.layer}.png"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to attn_maps directory")
    parser.add_argument("--layer", type=int, default=15, help="Target layer index")
    args = parser.parse_args()
    main(args)