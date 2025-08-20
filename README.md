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
  - FedPrune (static pruning)
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
git clone https://github.com/yourusername/FedTinyProp.git
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
model, accuracy, flops, memory, memory_saved, communication, sparsity, skipped_batches, \
    effective_compute_ratio, client_history, compression_ratio, quantization_error, \
    avg_scale_factor, history = federated_training(
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
# Dense baseline
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
    rigl_initial_sparsity=0.5,
    rigl_target_sparsity=0.95,
    rigl_update_interval=100,
    rigl_final_update_epoch=100
)
```

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

## RigL Integration

FedTinyProp also integrates RigL (Rigged Lottery) sparse training:

1. **Dynamic Connectivity**: Changes sparse connectivity patterns during training
2. **Momentum-Based Regrowth**: Uses gradient information for weight regrowth
3. **Cosine Sparsity Schedule**: Gradually increases sparsity during training

## Citation

If you use FedTinyProp in your research, please cite:

```bibtex
@article{fedtinyprop2023,
  title={FedTinyProp: Federated Learning with Adaptive Sparsity for Resource-Constrained Edge Devices},
  author={Author, A. and Author, B.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work builds upon research in federated learning, model compression, and sparse training
- Special thanks to contributors and the open-source community