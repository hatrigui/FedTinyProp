import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def plot_training_metrics(history: Dict, title_prefix: str = "", save_path: Optional[str] = None) -> None:
    """Plot training progress metrics."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = [12, 8]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Test Accuracy')
    ax1.set_title('Accuracy over Rounds')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend()
    
    # Plot sparsity
    ax2.plot(history['sparsity'], color='green')
    ax2.set_title('Sparsity over Rounds')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Sparsity')
    ax2.grid(True)
    
    # Plot memory usage
    memory_mb = np.array(history['memory_bytes']) / 1e6
    ax3.plot(memory_mb, color='red')
    ax3.set_title('Memory Usage')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Memory (MB)')
    ax3.grid(True)
    
    # Plot communication cost
    comm_mb = np.array(history['communication_bytes']) / 1e6
    ax4.plot(comm_mb, color='purple')
    ax4.set_title('Communication Cost')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Communication (MB)')
    ax4.grid(True)
    
    plt.suptitle(f"{title_prefix} Training Progress", y=1.02, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_efficiency_metrics(history: Dict, save_path: Optional[str] = None) -> None:
    """Plot efficiency metrics and trade-offs."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = [12, 8]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy vs sparsity trade-off
    scatter1 = ax1.scatter(history['sparsity'], history['accuracy'], 
                          c=np.arange(len(history['accuracy'])), cmap='viridis')
    ax1.set_title('Accuracy vs Sparsity Trade-off')
    ax1.set_xlabel('Sparsity')
    ax1.set_ylabel('Accuracy')
    plt.colorbar(scatter1, ax=ax1, label='Round')
    
    # Plot communication vs accuracy trade-off
    comm_mb = np.array(history['communication_bytes']) / 1e6
    scatter2 = ax2.scatter(comm_mb, history['accuracy'], 
                          c=history['sparsity'], cmap='viridis')
    ax2.set_title('Accuracy vs Communication Trade-off')
    ax2.set_xlabel('Communication (MB)')
    ax2.set_ylabel('Accuracy')
    plt.colorbar(scatter2, ax=ax2, label='Sparsity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_hardware_profile(metrics: Dict, hardware_constraints: Optional[Dict] = None, save_path: Optional[str] = None) -> None:
    """Plot hardware utilization metrics with optional constraints."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = [12, 8]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Memory usage
    memory_mb = np.array(metrics['memory_bytes']) / 1e6
    ax1.plot(memory_mb, color='blue', label='Usage')
    if hardware_constraints and 'memory_limit' in hardware_constraints:
        ax1.axhline(y=hardware_constraints['memory_limit']/1e6, color='b', linestyle='--', label='Limit')
    ax1.set_title('Memory Usage')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Memory (MB)')
    ax1.grid(True)
    ax1.legend()
    
    # Compute operations
    mflops = np.array(metrics['flops']) / 1e6
    ax2.plot(mflops, color='green', label='Usage')
    if hardware_constraints and 'compute_limit' in hardware_constraints:
        ax2.axhline(y=hardware_constraints['compute_limit']/1e6, color='g', linestyle='--', label='Limit')
    ax2.set_title('Compute Operations')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('MFLOPS')
    ax2.grid(True)
    ax2.legend()
    
    # Communication cost
    comm_mb = np.array(metrics['communication_bytes']) / 1e6
    ax3.plot(comm_mb, color='red', label='Usage')
    if hardware_constraints and 'communication_limit' in hardware_constraints:
        ax3.axhline(y=hardware_constraints['communication_limit']/1e6, color='r', linestyle='--', label='Limit')
    ax3.set_title('Communication Cost')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Communication (MB)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_convergence_analysis(history: Dict, save_path: Optional[str] = None) -> None:
    """Plot convergence analysis with sparsity impact."""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = [12, 8]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy convergence with sparsity
    ax1.plot(history['accuracy'], label='Accuracy')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(history['sparsity'], color='red', linestyle='--', label='Sparsity')
    ax1.set_title('Accuracy Convergence with Sparsity')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1_twin.set_ylabel('Sparsity', color='red')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Plot compute efficiency
    mflops = np.array(history['flops']) / 1e6
    ax2.plot(mflops, label='Compute Operations')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(history['sparsity'], color='red', linestyle='--', label='Sparsity')
    ax2.set_title('Compute Efficiency with Sparsity')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('MFLOPS')
    ax2_twin.set_ylabel('Sparsity', color='red')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show() 