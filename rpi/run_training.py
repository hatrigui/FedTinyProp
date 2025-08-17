#!/usr/bin/env python3
# FedTinyProp Raspberry Pi Training Script
# This script runs FedTinyProp training on CIFAR-10 with Dirichlet distribution (alpha=0.5)

import sys
import os
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import FedTinyProp modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FedTinyProp modules
from clients.federated_training import federated_training
from utils.data_partition import dirichlet_partition
from models.config import get_tinyprop_config, get_dense_config
from clients.aggregators import sparse_fedavg_aggregate
from torchvision import datasets, transforms

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rpi_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FedTinyProp-RPI-Training")

def load_cifar10():
    """Load CIFAR-10 dataset with memory-efficient transforms"""
    logger.info("Loading CIFAR-10 dataset...")
    
    # Memory-efficient transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10
    trainset = datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)

    logger.info(f"Training set size: {len(trainset)}")
    logger.info(f"Test set size: {len(testset)}")
    logger.info(f"Number of classes: {len(trainset.classes)}")
    
    return trainset, testset

def run_training(use_dense_baseline=False, rounds=100, batch_size=16, 
                alpha=0.5, num_clients=5, local_epochs=1):
    """Run FedTinyProp or Dense baseline training"""
    # Force CPU usage for Raspberry Pi
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get appropriate config
    if use_dense_baseline:
        config = get_dense_config('cifar10')
        model_name = "Dense"
    else:
        config = get_tinyprop_config('cifar10')
        model_name = "FedTinyProp"
    
    tinyprop_params = config["tinyprop_params"]
    
    # Adjust batch size for Raspberry Pi
    config["train_args"]["batch_size"] = batch_size
    
    logger.info(f"Starting {model_name} training...")
    logger.info(f"Configuration: rounds={rounds}, batch_size={batch_size}, "
               f"alpha={alpha}, num_clients={num_clients}, local_epochs={local_epochs}")
    
    # Load dataset
    trainset, testset = load_cifar10()
    
    # Partition data with Dirichlet distribution
    logger.info(f"Partitioning data with Dirichlet alpha={alpha}, num_clients={num_clients}...")
    partitions = dirichlet_partition(trainset, num_clients=num_clients, alpha=alpha)
    
    # Start timer
    start_time = time.time()
    
    # Run training
    model, metrics = federated_training(
        client_datasets=partitions,
        model_name='cifar10',
        testset=testset,
        tinyprop_params=tinyprop_params,
        aggregator_fn=sparse_fedavg_aggregate,
        rounds=rounds,
        device=device,
        local_epochs=local_epochs,
        early_stopping_patience=10,
        early_stopping_delta=0.001,
        csv_log_path=f'../results/cifar10_{model_name.lower()}_rpi_training.csv',
        initial_sparsity=tinyprop_params.S_min if not use_dense_baseline else 0.0,
        target_sparsity=tinyprop_params.S_max if not use_dense_baseline else 0.0,
        energy_budget=1000,
        save_dir='../results',
        save_interval=5,
        use_dense_baseline=use_dense_baseline,
        use_fedprox=False,
        use_rigl=False
    )
    
    # End timer
    training_time = time.time() - start_time
    
    # Print summary statistics
    logger.info(f"\nTraining Summary for {model_name}:")
    logger.info(f"Total Training Time: {training_time:.2f} seconds")
    logger.info(f"Final Accuracy: {metrics['accuracy'][-1]:.4f}")
    
    if not use_dense_baseline:
        logger.info(f"Average Sparsity: {np.mean(metrics['sparsity']):.4f}")
        logger.info(f"Average Compression Ratio: {np.mean(metrics['compression_ratio']):.4f}")
        logger.info(f"Average Effective Compute Ratio: {np.mean(metrics['effective_compute_ratio']):.4f}")
    
    logger.info(f"Total Communication Cost: {metrics['communication'][-1]/1024:.2f} KB")
    logger.info(f"Total Memory Used: {metrics['memory'][-1]/1024/1024:.2f} MB")
    
    if not use_dense_baseline and 'memory_saved' in metrics:
        logger.info(f"Total Memory Saved: {metrics['memory_saved'][-1]/1024/1024:.2f} MB")
    
    # Save model
    model_path = f'../results/cifar10_{model_name.lower()}_rpi_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, metrics, training_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedTinyProp Raspberry Pi Training')
    parser.add_argument('--dense', action='store_true', help='Use dense baseline model')
    parser.add_argument('--rounds', type=int, default=100, help='Number of training rounds')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha parameter')
    parser.add_argument('--num-clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--local-epochs', type=int, default=1, help='Number of local epochs')
    
    args = parser.parse_args()
    
    # Run training
    model, metrics, training_time = run_training(
        use_dense_baseline=args.dense,
        rounds=args.rounds,
        batch_size=args.batch_size,
        alpha=args.alpha,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs
    )
