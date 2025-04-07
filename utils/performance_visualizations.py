import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_fedtinyprop_metrics(csv_path, title_prefix=""):
    df = pd.read_csv(csv_path)
    rounds = df["round"]
    
    # Define available metrics and their properties
    metrics = [
        ('accuracy', 'Accuracy', 'blue', None),
        ('sparsity', 'Sparsity', 'green', None),
        ('memory_bytes', 'Memory (MB)', 'blue', lambda x: x / 1e6),
        ('flops', 'FLOPS (M)', 'orange', lambda x: x / 1e6),
        ('communication_bytes', 'Communication (MB)', 'red', lambda x: x / 1e6),
        ('avg_grad_norm', 'Gradient Norm', 'purple', None),
        ('avg_phi', 'Phi', 'teal', None),
        ('effective_compute_ratio', 'Effective Compute Ratio', 'brown', None),
        ('compression_ratio', 'Compression Ratio', 'pink', None)
    ]
    
    # Filter out metrics that don't exist in the dataframe
    available_metrics = [(m, t, c, f) for m, t, c, f in metrics if m in df.columns]
    
    # Calculate number of rows needed
    n_metrics = len(available_metrics)
    n_rows = (n_metrics + 1) // 2  # Round up division
    
    plt.figure(figsize=(16, 4 * n_rows))
    
    # Plot each available metric
    for idx, (metric, title, color, transform) in enumerate(available_metrics):
        plt.subplot(n_rows, 2, idx + 1)
        
        # Apply transformation if specified
        values = df[metric]
        if transform:
            values = transform(values)
        
        plt.plot(rounds, values, marker="o", color=color, label=title)
        plt.title(f"{title_prefix} - {title}")
        plt.xlabel("Round")
        plt.ylabel(title)
        plt.grid(True)
        
        # Add legend for metrics that might have multiple lines
        if metric in ['avg_grad_norm', 'avg_phi']:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_metric_comparison_across_partitions(csv_paths_dict, metric="accuracy", title=None, ylabel=None):
    plt.figure(figsize=(12, 8))
    
    for partition_name, csv_path in csv_paths_dict.items():
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            print(f"[WARN] '{metric}' not found in {csv_path}, skipping...")
            continue
        
        # Scale metrics appropriately
        if metric == "memory_bytes":
            values = df[metric] / 1e6  # Convert to MB
            ylabel = "Memory (MB)"
        elif metric == "flops":
            values = df[metric] / 1e6  # Convert to MFLOPs
            ylabel = "FLOPS (M)"
        elif metric == "communication_bytes":
            values = df[metric] / 1e6  # Convert to MB
            ylabel = "Communication (MB)"
        else:
            values = df[metric]
        
        plt.plot(df["round"], values, label=partition_name, marker='o', linewidth=2)

    plt.xlabel("Round")
    plt.ylabel(ylabel or metric.replace("_", " ").title())
    plt.title(title or f"Comparison of {metric.replace('_', ' ').title()} Across Partitions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hardware_efficiency(csv_paths_dict):
    """Plot hardware efficiency metrics across partitions."""
    # Define available metrics and their properties
    metrics = {
        "memory_bytes": ("Memory Efficiency (MB)", lambda x: x / 1e6),
        "flops": ("Compute Efficiency (MFLOPs)", lambda x: x / 1e6),
        "communication_bytes": ("Communication Efficiency (MB)", lambda x: x / 1e6)
    }
    
    # Filter out metrics that don't exist in the first CSV
    first_csv = next(iter(csv_paths_dict.values()))
    df = pd.read_csv(first_csv)
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    
    if not available_metrics:
        print("No hardware metrics available in the CSV files")
        return
    
    n_metrics = len(available_metrics)
    n_rows = (n_metrics + 1) // 2  # Round up division
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    if n_metrics == 1:
        axes = [[axes]]
    axes = axes.flatten()
    
    for idx, (metric, (title, transform)) in enumerate(available_metrics.items()):
        for partition_name, csv_path in csv_paths_dict.items():
            df = pd.read_csv(csv_path)
            if metric not in df.columns:
                continue
                
            values = transform(df[metric])
            axes[idx].plot(df["round"], values, label=partition_name, marker='o')
        
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Round")
        axes[idx].set_ylabel(title.split("(")[1].strip(")"))
        axes[idx].grid(True)
        axes[idx].legend()
    
    # Hide any unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

