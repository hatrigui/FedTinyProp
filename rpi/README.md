# FedTinyProp for Raspberry Pi

This directory contains scripts and utilities for running FedTinyProp on a Raspberry Pi device, specifically for training CIFAR-10 with a Dirichlet distribution (alpha=0.5) and benchmarking against a dense model.

## Setup Instructions

### 1. Prepare Raspberry Pi

First, ensure your Raspberry Pi is properly set up:

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    libatlas-base-dev \
    gfortran \
    git \
    htop \
    iotop
```

### 2. Clone Repository

Clone the FedTinyProp repository:

```bash
git clone https://github.com/your-username/FedTinyProp.git
cd FedTinyProp
```

### 3. Set Up Environment

Run the setup script to create a virtual environment and install dependencies:

```bash
# Make the setup script executable
chmod +x rpi/setup.sh

# Run the setup script
./rpi/setup.sh

# Activate the virtual environment
source fedtinyprop_env/bin/activate
```

Alternatively, you can install dependencies manually:

```bash
# Create and activate virtual environment
python3 -m venv fedtinyprop_env
source fedtinyprop_env/bin/activate

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r rpi/requirements.txt
```

## Available Scripts

### Verify Model Configuration

Before running benchmarks, verify that the FedTinyProp model is properly sparsified, pruned, and compressed:

```bash
python rpi/verify_model.py --model both
```

This will check both FedTinyProp and dense models and provide a comparison of their configurations.

### Run Training

To run FedTinyProp training on CIFAR-10:

```bash
# Run FedTinyProp training
python rpi/run_training.py

# Run dense baseline training
python rpi/run_training.py --dense

# Customize parameters
python rpi/run_training.py --rounds 50 --batch-size 16 --alpha 0.5 --num-clients 5 --local-epochs 1
```

### Run Benchmarks

To run comprehensive benchmarks comparing FedTinyProp and dense models:

```bash
# Run full benchmarks
python rpi/benchmark.py

# Customize benchmark parameters
python rpi/benchmark.py --rounds 20 --batch-size 16 --alpha 0.5 --num-clients 5

# Only plot existing results
python rpi/benchmark.py --plot-only
```

### Monitor System Resources

To monitor system resources during training:

```bash
# Start monitoring in a separate terminal
python rpi/monitor.py

# Monitor for a specific duration (in seconds)
python rpi/monitor.py --duration 3600

# Monitor with custom interval and output file
python rpi/monitor.py --interval 2.0 --output my_metrics.csv --plot
```

### Analyze Results

After running benchmarks, analyze the results:

```bash
# Analyze and visualize benchmark results
python rpi/analyze_results.py

# Specify custom results file and plots directory
python rpi/analyze_results.py --results results/rpi_benchmark_results.json --plots-dir results/my_plots

# Generate summary report
python rpi/analyze_results.py --summary
```

## Benchmarking Metrics

The benchmarking scripts collect and compare the following metrics:

1. **Accuracy**: Test accuracy of the trained models
2. **Training Time**: Total time required for training
3. **Memory Usage**: Memory consumption during training
4. **Communication Cost**: Total bytes transferred during federated training
5. **FLOPs**: Floating point operations for model inference
6. **Model Size**: Size of the model in memory
7. **Sparsity**: Percentage of zero parameters in the model
8. **Compression Ratio**: Ratio of dense model size to compressed model size
9. **System Metrics**: CPU usage, memory usage, and temperature during training

## Raspberry Pi Optimization Tips

1. **Memory Management**: 
   - Use smaller batch sizes (8-16) to reduce memory usage
   - Enable garbage collection between rounds
   - Monitor memory usage with `rpi/monitor.py`

2. **Temperature Control**:
   - Use a cooling fan or heatsink to prevent throttling
   - Monitor temperature with `rpi/monitor.py`
   - Consider adding breaks between training rounds if temperature gets too high

3. **Power Supply**:
   - Use a stable power supply (5V/3A recommended)
   - Avoid running on battery power

4. **Storage**:
   - Ensure at least 2GB of free space for dataset and results
   - Use an external SSD if available for better performance

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce batch size (try 8 or 4)
2. Reduce number of clients
3. Increase model sparsity
4. Disable data augmentation in transforms

### Slow Training

If training is too slow:

1. Reduce number of rounds
2. Reduce number of local epochs
3. Ensure you're using the sparsified FedTinyProp model
4. Check for background processes consuming CPU

### Overheating

If the Raspberry Pi is overheating:

1. Add cooling (fan, heatsink)
2. Add breaks between rounds
3. Reduce CPU frequency:
   ```bash
   sudo cpufreq-set -u 1000MHz
   ```

## Results Directory Structure

After running benchmarks, results will be saved in the following structure:

```
results/
├── cifar10_fedtinyprop_rpi_metrics.csv    # FedTinyProp training metrics
├── cifar10_dense_rpi_metrics.csv          # Dense model training metrics
├── cifar10_fedtinyprop_rpi_system_metrics.csv  # System metrics during FedTinyProp training
├── cifar10_dense_rpi_system_metrics.csv    # System metrics during dense training
├── rpi_benchmark_results.json             # Benchmark comparison results
├── benchmark_comparison.csv               # CSV comparison of metrics
├── benchmark_summary.md                   # Markdown summary report
└── plots/                                 # Visualization plots
    ├── accuracy_curves.png
    ├── memory_usage.png
    ├── communication_cost.png
    ├── flops.png
    ├── sparsity.png
    ├── cpu_usage.png
    ├── memory_usage_time.png
    └── temperature.png
```
