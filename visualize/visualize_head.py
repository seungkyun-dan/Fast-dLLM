import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_layer_data

def main(args):
    data = load_layer_data(args.dir, args.step, args.layer)
    if data is None:
        print("Data not found.")
        return

    # Select specific head: (Q, K)
    attn_map = data[args.head].numpy()

    plt.figure(figsize=(10, 10))
    sns.heatmap(attn_map, cmap="Reds", square=True, cbar=False)
    
    plt.title(f"Attention Map: Step {args.step} | L{args.layer} | H{args.head}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    
    out_path = f"heatmap_s{args.step}_l{args.layer}_h{args.head}.png"
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to attn_maps directory")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--head", type=int, default=0)
    args = parser.parse_args()
    main(args)