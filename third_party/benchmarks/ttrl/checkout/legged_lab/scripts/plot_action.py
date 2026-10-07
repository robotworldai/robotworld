import argparse
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot recorded play data (obs[48:69]).")
    parser.add_argument("--npz", type=str, required=True, help="Path to play_obs_actions.npz saved under logs.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path. Defaults next to NPZ.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    args = parser.parse_args()

    npz_path = os.path.abspath(args.npz)
    if not os.path.isfile(npz_path):
        print(f"[ERROR] NPZ file not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path)

    # Prefer precomputed slice; fall back to computing from obs
    if "obs_slice" in data and data["obs_slice"].size > 0:
        slice_arr = data["obs_slice"]
    elif "obs" in data and data["obs"].size > 0:
        obs_arr = data["obs"]
        if obs_arr.ndim >= 2 and obs_arr.shape[1] >= 69:
            slice_arr = obs_arr[:, 48:69]
        else:
            print("[ERROR] 'obs' in NPZ is too small to slice [48:69].")
            sys.exit(2)
    else:
        print("[ERROR] No 'obs_slice' or 'obs' found with data in NPZ.")
        sys.exit(3)

    # Determine output path
    if args.output is None:
        out_dir = os.path.dirname(npz_path)
        out_path = os.path.join(out_dir, "obs_48_69_plot.png")
    else:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(slice_arr)
    plt.title("obs[48:69] over time (env 0)")
    plt.xlabel("step")
    plt.ylabel("value")
    if slice_arr.ndim == 2 and slice_arr.shape[1] > 1:
        plt.legend([f"d{i}" for i in range(slice_arr.shape[1])], ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    if args.show:
        plt.show()
    plt.close()

    print(f"[INFO] Saved plot to: {out_path}")


if __name__ == "__main__":
    main()

