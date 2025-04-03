import pandas as pd
import matplotlib.pyplot as plt

def plot_fedtinyprop_metrics(csv_path, title_prefix=""):
    df = pd.read_csv(csv_path)
    rounds = df["round"]

    plt.figure(figsize=(16, 20))

    # 1. Accuracy
    plt.subplot(4, 2, 1)
    plt.plot(rounds, df["accuracy"], marker="o", label="Accuracy")
    plt.title(f"{title_prefix} - Test Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # 2. Sparsity
    plt.subplot(4, 2, 2)
    plt.plot(rounds, df["sparsity"], marker="o", color="green", label="Sparsity")
    plt.title(f"{title_prefix} - Model Update Sparsity")
    plt.xlabel("Round")
    plt.ylabel("Sparsity")
    plt.grid(True)

    # 3. Effective Compute Ratio
    plt.subplot(4, 2, 3)
    plt.plot(rounds, df["effective_compute_ratio"], marker="o", color="purple", label="Effective Compute Ratio")
    plt.title(f"{title_prefix} - Effective Computation")
    plt.xlabel("Round")
    plt.ylabel("Compute Ratio")
    plt.grid(True)

    # 4. Skipped Batches
    plt.subplot(4, 2, 4)
    plt.plot(rounds, df["skipped_batches"], marker="o", color="orange", label="Skipped Batches")
    plt.title(f"{title_prefix} - Skipped Batches per Round")
    plt.xlabel("Round")
    plt.ylabel("Skipped Batches")
    plt.grid(True)

    # 5. Communication Cost
    plt.subplot(4, 2, 5)
    plt.plot(rounds, df["communication_bytes"] / 1e6, marker="o", color="red", label="Comm MB")
    plt.title(f"{title_prefix} - Communication Cost (MB)")
    plt.xlabel("Round")
    plt.ylabel("MB Sent")
    plt.grid(True)

    # 6. Compression Ratio
    plt.subplot(4, 2, 6)
    plt.plot(rounds, df["compression_ratio"], marker="o", color="teal", label="Compression Ratio")
    plt.title(f"{title_prefix} - Delta Compression Ratio")
    plt.xlabel("Round")
    plt.ylabel("Ratio")
    plt.grid(True)

    # 7. Gradient Norm & Phi
    plt.subplot(4, 2, 7)
    plt.plot(rounds, df["avg_grad_norm"], label="Avg Grad Norm", marker="o")
    plt.plot(rounds, df["avg_phi"], label="Avg Phi", marker="x")
    plt.title(f"{title_prefix} - Grad Norm & Adaptive Ratio (Phi)")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_comparison_across_partitions(csv_paths_dict, metric="accuracy", title=None, ylabel=None):
   
    plt.figure(figsize=(10, 6))

    for partition_name, csv_path in csv_paths_dict.items():
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            print(f"[WARN] '{metric}' not found in {csv_path}, skipping...")
            continue
        plt.plot(df["round"], df[metric], label=partition_name, marker='o', linewidth=2)

    plt.xlabel("Round")
    plt.ylabel(ylabel or metric.replace("_", " ").title())
    plt.title(title or f"Comparison of {metric.replace('_', ' ').title()} Across Partitions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

