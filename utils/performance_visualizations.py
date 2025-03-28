
import matplotlib.pyplot as plt

def plot_fed_metrics(acc_list, flops_list, mem_list, comm_list, sparsity_list):
    
    rounds = range(1, len(acc_list)+1)

    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.plot(rounds, acc_list, label='Accuracy')
    plt.title("Accuracy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")

    plt.subplot(2,3,2)
    plt.plot(rounds, flops_list, label='FLOPs', color='orange')
    plt.title("FLOPs vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Approx FLOPs")

    plt.subplot(2,3,3)
    plt.plot(rounds, mem_list, label='Mem Usage', color='green')
    plt.title("Memory vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Bytes")

    plt.subplot(2,3,4)
    plt.plot(rounds, comm_list, label='Comm Bytes', color='red')
    plt.title("Comm Cost vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Bytes")

    plt.subplot(2,3,5)
    plt.plot(rounds, [s*100 for s in sparsity_list], label='Sparsity %', color='purple')
    plt.title("Sparsity vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Non-zero fraction (%)")

    plt.tight_layout()
    plt.show()


def plot_all_strategies(all_results):
    for strategy, metrics in all_results.items():
        print(f"\nPlotting results for partition strategy: {strategy.upper()}")
        acc = metrics["acc"]
        flops = metrics["flops"]
        mem = metrics["mem"]
        comm = metrics["comm"]
        sparsity = metrics["sparsity"]

        final_acc     = acc[-1] if acc else 0
        final_flops   = flops[-1] if flops else 0
        final_mem     = mem[-1] if mem else 0
        final_comm    = comm[-1] if comm else 0
        final_spars   = sparsity[-1] * 100 if sparsity else 0
        print(f"Final Accuracy: {final_acc:.4f}, FLOPs: {final_flops:.2f}, Mem: {final_mem:.2f} bytes, Comm: {final_comm:.2f} bytes, Sparsity: {final_spars:.2f}%")

        plt.figure(figsize=(12,8))
        plt.suptitle(f"{strategy.upper()} - Federated Training Metrics", fontsize=16)

        plot_fed_metrics(acc, flops, mem, comm, sparsity)
        

def plot_all_partitions_in_one_figure(all_results):


    metric_keys = ["acc", "flops", "mem", "comm", "sparsity"]
    metric_titles = ["Accuracy", "FLOPs", "Memory", "Communication", "Sparsity"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Federated Training Metrics by Partition Strategy", fontsize=16)
    axs_flat = axs.ravel()

    for i, key in enumerate(metric_keys):
        ax = axs_flat[i]  
        ax.set_title(metric_titles[i])
        for strategy, data in all_results.items():
            if key not in data:
                continue 

            y_values = data[key]
            rounds_axis = range(1, len(y_values) + 1)

            if key == "sparsity":
                y_values = [val * 100 for val in y_values]
            ax.plot(rounds_axis, y_values, label=strategy)

        ax.set_xlabel("Round")
        if key == "acc":
            ax.set_ylabel("Accuracy")
        elif key == "flops":
            ax.set_ylabel("Approx FLOPs")
        elif key == "mem":
            ax.set_ylabel("Memory (bytes)")
        elif key == "comm":
            ax.set_ylabel("Comm (bytes)")
        elif key == "sparsity":
            ax.set_ylabel("Non-zero fraction (%)")

        ax.legend()

    axs_flat[-1].axis("off")

    plt.tight_layout()
    plt.show()

