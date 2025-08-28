# FedTinyProp: Adaptive Sparse Backpropagation for Efficient Federated Learning on Embedded Devices

FedTinyProp is a federated learning framework designed for resource-constrained edge devices, featuring adaptive sparsity, gradient quantization, and batch skipping mechanisms to reduce computation, memory, and communication costs.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

## Features

- **Adaptive Sparsity**: Dynamically adjusts sparsity levels based on gradient norms
- **Gradient Quantization**: Reduces communication costs with configurable bit precision
- **Batch Skipping**: Intelligently skips unnecessary computation for efficiency
- **Multiple Training Methods**:
  - FedTinyProp (adaptive sparsity)
  - Dense (baseline)
  - FedProx (with regularization)
  - RigL (dynamic sparse training)
- **Comprehensive Metrics**: Tracks accuracy, FLOPs, memory usage, communication costs, sparsity, and more
- **Raspberry Pi Optimization**: Specialized support for deployment on Raspberry Pi devices

## Project Structure

```
├── clients/                # Client implementations for federated learning
│   ├── aggregators.py      # Aggregation functions for client updates
│   ├── federated_client.py # Core federated client implementation
│   ├── federated_training.py # Main federated training loop
│   └── rigl_client.py      # RigL sparse training client
├── datasets/               # Dataset implementations
│   ├── har_dataset.py      # Human Activity Recognition dataset
│   └── speech_commands_dataset.py # Speech Commands dataset
├── models/                 # Model implementations
│   ├── config.py           # Model configuration
│   ├── model.py            # Model architecture definitions
│   ├── rigl.py             # RigL sparse training implementation
│   └── tinyProp.py         # TinyProp adaptive sparsity layers
├── notebooks/              # Jupyter notebooks for experiments
├── rpi/                    # Raspberry Pi specific code and instructions
└── utils/                  # Utility functions
    ├── adaptive_sparsification.py # Adaptive sparsity utilities
    ├── data_partition.py   # Data partitioning for federated learning
    ├── early_stopping.py   # Early stopping implementation
    └── flops_calculator.py # FLOPs calculation utilities
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/hatrigui/FedTinyProp
cd FedTinyProp

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Federated Model

```python
from clients.federated_training import federated_training
from utils.data_partition import create_federated_datasets
from clients.aggregators import sparse_fedavg_aggregate
import torch

# Create federated datasets
client_datasets, test_dataset = create_federated_datasets(
    dataset_name="cifar10",
    num_clients=10,
    partition_type="iid"
)

# Define TinyProp parameters
tinyprop_params = {
    "S_min": 0.3,  # Minimum sparsity
    "S_max": 0.9,  # Maximum sparsity
    "zeta": 0.3,   # Sparsity scaling factor
    "phi_min": 0.01  # Minimum gradient norm threshold
}

# Train the model
model, metrics = federated_training(
    client_datasets=client_datasets,
    model_name="cifar10",
    testset=test_dataset,
    tinyprop_params=tinyprop_params,
    aggregator_fn=sparse_fedavg_aggregate,
    rounds=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
    local_epochs=5,
    quantization_bits=8,
    save_dir="./results"
)
```

### Using Different Training Methods

```python
# Dense baseline (disables FedTinyProp)
model, metrics = federated_training(
    # ... other parameters ...
    use_dense_baseline=True
)

# FedProx with regularization
model, metrics = federated_training(
    # ... other parameters ...
    use_fedprox=True,
    fedprox_mu=0.1  # Regularization parameter
)

# RigL dynamic sparse training
model, metrics = federated_training(
    # ... other parameters ...
    use_rigl=True,
    rigl_initial_sparsity=0.5,  # Initial sparsity level
    rigl_target_sparsity=0.95,  # Final target sparsity
    rigl_update_interval=100,    # How often to update masks
    rigl_final_update_epoch=100  # When to stop updating masks
)
```

### Example Training Loop with Multiple Benchmarks

The following example shows how to train and analyze a model with different data partitioning strategies:

```python
from clients.federated_training import federated_training
from clients.aggregators import sparse_fedavg_aggregate
from utils.data_partition import create_federated_datasets
import torch
import numpy as np

def train_and_analyze_partition(partition_name, client_datasets, tinyprop_params):
    print(f"\nTraining on partition: {partition_name.upper()}")
    
    # Run training
    model, metrics = federated_training(
        client_datasets=client_datasets,
        model_name='speechcommands',
        testset=test_dataset,
        tinyprop_params=tinyprop_params,
        aggregator_fn=sparse_fedavg_aggregate,
        rounds=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_epochs=1,
        early_stopping_patience=100,  # Stop if no improvement for 100 rounds
        early_stopping_delta=0.001,   # Minimum improvement to continue
        csv_log_path=f'results/speechcommands_metrics.csv',
        initial_sparsity=tinyprop_params.S_min,
        target_sparsity=tinyprop_params.S_max,
        energy_budget=1000,           # Energy constraint for devices
        save_dir='results',
        save_interval=1,               # Save model every N rounds
        
        # Training method flags (only one should be True at a time)
        use_dense_baseline=False,      # Dense baseline (no sparsity)
        use_fedprox=False,             # FedProx regularization
        fedprox_mu=0.01,               # FedProx regularization strength
        use_rigl=True,                 # RigL dynamic sparse training
        rigl_initial_sparsity=0.5,     # Initial RigL sparsity
        rigl_target_sparsity=0.95,     # Target RigL sparsity
        rigl_update_interval=100,      # How often to update masks
        rigl_final_update_epoch=100,   # When to stop updating masks
    )
```

### Notebooks for Different Datasets

The `notebooks/` directory contains Jupyter notebooks for experimenting with different datasets:

- `SpeechCommands_5cl.ipynb`: Speech Commands dataset with 5 clients
- `cifar10.ipynb`: CIFAR-10 image classification
- `cifar100_5cl.ipynb`: CIFAR-100 with 5 clients
- `fashionmnist_5cl.ipynb`: Fashion MNIST with 5 clients
- And more...

Each notebook contains dataset-specific preprocessing and training loops. You can use these notebooks as templates for your own experiments.

### Key Parameters

- **Training Method Flags**: Set only one to `True` at a time
  - `use_dense_baseline=True`: Standard training without sparsity (baseline)
  - `use_fedprox=True`: FedProx with regularization
  - `use_rigl=True`: RigL dynamic sparse training
  - All `False`: Default FedTinyProp with adaptive sparsity

- **TinyProp Parameters**:
  - `S_min`: Minimum sparsity level (0.0-1.0)
  - `S_max`: Maximum sparsity level (0.0-1.0)
  - `zeta`: Sparsity scaling factor
  - `phi_min`: Minimum gradient norm threshold

- **RigL Parameters**:
  - `rigl_initial_sparsity`: Starting sparsity level
  - `rigl_target_sparsity`: Final target sparsity level
  - `rigl_update_interval`: How often to update masks (iterations)
  - `rigl_final_update_epoch`: When to stop updating masks

## Raspberry Pi Deployment

For detailed instructions on deploying and benchmarking FedTinyProp on Raspberry Pi devices, please refer to the [Raspberry Pi README](./rpi/README.md).

## Metrics and Evaluation

FedTinyProp tracks comprehensive metrics during training:

- **Accuracy**: Model accuracy on test data
- **FLOPs**: Floating-point operations per second
- **Memory**: Peak memory usage
- **Memory Saved**: Memory savings from sparsity
- **Communication**: Total bytes transferred
- **Sparsity**: Average model sparsity
- **Skipped Batches**: Number of computation-saving batch skips
- **Effective Compute Ratio**: Ratio of actual to potential computation
- **Compression Ratio**: Communication compression achieved
- **Quantization Error**: Error introduced by quantization

## TinyProp Adaptive Sparsity

The core innovation in FedTinyProp is its adaptive sparsity mechanism:

1. **Gradient Norm Tracking**: Monitors gradient magnitudes to determine importance
2. **Phi Parameter**: Adaptively adjusts sparsity threshold based on gradient norms
3. **Batch Skipping**: Intelligently skips batches when gradients are below threshold
4. **Top-K Selection**: Selects only the most important gradient elements for updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work builds upon research in federated learning, model compression, and sparse training
- Special thanks to contributors and the open-source community