#!/usr/bin/env python3
# FedTinyProp Raspberry Pi Benchmarking Script
# This script runs benchmarks comparing FedTinyProp vs Dense model on CIFAR-10

import sys
import os
import time
import gc
import json
import numpy as np
import torch
import torch.nn 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path to import FedTinyProp modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FedTinyProp modules
from clients.federated_training import federated_training
from utils.data_partition import dirichlet_partition
from models.config import get_tinyprop_config, get_dense_config
from clients.aggregators import sparse_fedavg_aggregate
from utils.flops_calculator import compute_model_flops
from models.model import get_tinyprop_model

# Force CPU usage for Raspberry Pi
device = "cpu"
torch.manual_seed(42)
np.random.seed(42)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rpi_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FedTinyProp-RPI")

# Raspberry Pi specific monitoring functions
def get_raspberry_pi_memory_usage():
    """Get memory usage specific to Raspberry Pi"""
    try:
        # Get memory information from /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.readlines()
        
        # Parse memory information
        mem_total = None
        mem_free = None
        mem_available = None
        mem_buffers = None
        mem_cached = None
        sram_total = None
        sram_used = None
        
        for line in mem_info:
            if 'MemTotal' in line:
                mem_total = int(line.split()[1]) / 1024  # Convert to MB
            elif 'MemFree' in line:
                mem_free = int(line.split()[1]) / 1024  # Convert to MB
            elif 'MemAvailable' in line:
                mem_available = int(line.split()[1]) / 1024  # Convert to MB
            elif 'Buffers' in line:
                mem_buffers = int(line.split()[1]) / 1024  # Convert to MB
            elif 'Cached' in line:
                mem_cached = int(line.split()[1]) / 1024  # Convert to MB
        
        # Try to get SRAM usage (specific to Raspberry Pi)
        try:
            # Check if vcgencmd is available
            import subprocess
            sram_output = subprocess.check_output(['vcgencmd', 'get_mem', 'arm']).decode('utf-8')
            if sram_output and '=' in sram_output:
                sram_total = int(sram_output.split('=')[1].strip('M\n'))
                
            # Estimate SRAM usage based on process memory
            process = psutil.Process(os.getpid())
            sram_used = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.debug(f"Could not get SRAM info: {str(e)}")
            sram_total = None
            sram_used = None
        
        # Calculate used memory
        if mem_total is not None and mem_available is not None:
            mem_used = mem_total - mem_available
            mem_percent = (mem_used / mem_total) * 100
            return {
                'total_mb': mem_total,
                'used_mb': mem_used,
                'free_mb': mem_available,
                'percent_used': mem_percent,
                'sram_total_mb': sram_total,
                'sram_used_mb': sram_used
            }
        else:
            mem_info = psutil.virtual_memory()._asdict()
            mem_info['sram_total_mb'] = sram_total
            mem_info['sram_used_mb'] = sram_used
            return mem_info
    except Exception as e:
        logger.error(f"Error getting Raspberry Pi memory: {str(e)}")
        return psutil.virtual_memory()._asdict()

def get_raspberry_pi_temperature():
    """Get CPU temperature of Raspberry Pi"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
        return temp
    except Exception as e:
        logger.error(f"Error getting Raspberry Pi temperature: {str(e)}")
        return None

def check_throttling():
    """Check if Raspberry Pi is throttling due to temperature"""
    # Return default values without attempting to read files
    # This avoids error messages on systems where throttling files don't exist
    return {
        'under_voltage_detected': False,
        'frequency_capped': False,
        'throttling_active': False,
        'under_voltage_occurred': False,
        'frequency_capping_occurred': False,
        'throttling_occurred': False
    }

# Function to measure execution time
def measure_time(func, *args, **kwargs):
    """Measure execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

class BenchmarkMonitor:
    """Monitor system metrics during benchmarking"""
    def __init__(self, interval=1.0, model=None):
        self.interval = interval
        self.running = False
        self.model = model
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'sram_mb': [],
            'temperature': [],
            'timestamp': [],
            'latency_ms': []
        }
        
    def start(self):
        """Start monitoring in a separate thread"""
        import threading
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        
    def _monitor(self):
        """Monitor system metrics"""
        while self.running:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Get memory usage
                memory = get_raspberry_pi_memory_usage()
                
                # Get temperature
                temp = get_raspberry_pi_temperature()
                
                # Measure latency (time to process a small batch)
                latency = self._measure_processing_latency()
                
                # Record metrics
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_percent'].append(memory.get('percent_used', 0))
                self.metrics['memory_mb'].append(memory.get('used_mb', 0))
                self.metrics['sram_mb'].append(memory.get('sram_used_mb', 0))
                self.metrics['temperature'].append(temp)
                self.metrics['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                self.metrics['latency_ms'].append(latency)
            except Exception as e:
                logger.error(f"Error monitoring system: {str(e)}")
            
            # Sleep for the specified interval
            time.sleep(self.interval)
            
    def _measure_processing_latency(self):
        """Measure processing latency in milliseconds"""
        try:
            # Create a small tensor and measure processing time
            if self.model is not None:
                device = next(self.model.parameters()).device
                # Create a small batch (1x3x32x32) for CIFAR-10
                dummy_input = torch.rand(1, 3, 32, 32).to(device)
                
                # Warm-up
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                # Measure inference time
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                end_time = time.time()
                
                # Convert to milliseconds
                return (end_time - start_time) * 1000
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"Error measuring latency: {str(e)}")
            return 0.0
    
    def get_summary(self):
        """Get summary of monitored metrics"""
        if not self.metrics['cpu_percent']:
            return {}
        
        return {
            'avg_cpu_percent': np.mean(self.metrics['cpu_percent']),
            'max_cpu_percent': np.max(self.metrics['cpu_percent']),
            'avg_memory_mb': np.mean(self.metrics['memory_mb']),
            'max_memory_mb': np.max(self.metrics['memory_mb']),
            'avg_temperature': np.mean(self.metrics['temperature']) if self.metrics['temperature'] else None,
            'max_temperature': np.max(self.metrics['temperature']) if self.metrics['temperature'] else None,
            'duration_seconds': self.metrics['timestamp'][-1] - self.metrics['timestamp'][0]
        }
    
    def save_metrics(self, filename):
        """Save metrics to CSV file"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filename, index=False)
        logger.info(f"Saved monitoring metrics to {filename}")

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

def partition_data(trainset, alpha=0.5, num_clients=5):
    """Partition data with Dirichlet distribution"""
    logger.info(f"Partitioning data with Dirichlet alpha={alpha}, num_clients={num_clients}...")
    partitions = dirichlet_partition(trainset, num_clients=num_clients, alpha=alpha)
    return partitions

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")
            raise

def benchmark_model(partitions, testset, use_dense_baseline=False, rounds=20, batch_size=16):
    """Run benchmarking for a model (FedTinyProp or Dense)"""
    # Create results directory
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    ensure_dir_exists(results_dir)
    
    # Get appropriate config
    if use_dense_baseline:
        config = get_dense_config('cifar10')
        model_name = "Dense"
    else:
        config = get_tinyprop_config('cifar10')
        model_name = "FedTinyProp"
    
    tinyprop_params = config["tinyprop_params"]
    
    # Ensure train_args exists in config and adjust batch size for Raspberry Pi
    if "train_args" not in config:
        config["train_args"] = {}
    config["train_args"]["batch_size"] = batch_size
    
    logger.info(f"Starting {model_name} training for {rounds} rounds with batch size {batch_size}...")
    
    # Create model first
    tinyprop_params = config.get('tinyprop_params', {})
    global_model = get_tinyprop_model('cifar10', tinyprop_params).to(device)
    
    # Start monitoring
    monitor = BenchmarkMonitor(interval=1.0, model=global_model)
    monitor.start()
    
    try:
        # Measure training time
        (model, metrics), training_time = measure_time(
            federated_training,
            client_datasets=partitions,
            model_name='cifar10',
            testset=testset,
            tinyprop_params=tinyprop_params,
            aggregator_fn=sparse_fedavg_aggregate,
            rounds=rounds,
            device=device,
            local_epochs=1,
            early_stopping_patience=5,
            early_stopping_delta=0.001,
            csv_log_path=os.path.join(results_dir, f'cifar10_{model_name.lower()}_rpi_metrics.csv'),
            initial_sparsity=tinyprop_params.S_min if not use_dense_baseline else 0.0,
            target_sparsity=tinyprop_params.S_max if not use_dense_baseline else 0.0,
            energy_budget=500,
            save_dir=results_dir,
            save_interval=1,
            use_dense_baseline=True,
            use_fedprox=False,
            use_rigl=False
        )
        
        # Stop monitoring
        monitor.stop()
        
        # Save monitoring metrics
        monitor.save_metrics(os.path.join(results_dir, f'cifar10_{model_name.lower()}_rpi_system_metrics.csv'))
        
        # Get monitoring summary
        monitoring_summary = monitor.get_summary()
        
        # Add SRAM and latency metrics to the training metrics
        if 'sram_usage' not in metrics:
            metrics['sram_usage'] = []
        if 'latency_ms' not in metrics:
            metrics['latency_ms'] = []
            
        # Add average SRAM and latency from monitoring
        avg_sram = np.mean(monitor.metrics['sram_mb']) if monitor.metrics['sram_mb'] else 0
        avg_latency = np.mean(monitor.metrics['latency_ms']) if monitor.metrics['latency_ms'] else 0
        
        # Fill metrics with the same length as other metrics
        for _ in range(len(metrics['round'])):
            metrics['sram_usage'].append(avg_sram)
            metrics['latency_ms'].append(avg_latency)
        
        # Calculate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Calculate FLOPs
        flops_dict = compute_model_flops(model)
        total_flops = sum(flops_dict.values())
        
        # Print summary statistics
        logger.info(f"\nTraining Summary for {model_name}:")
        logger.info(f"Total Training Time: {training_time:.2f} seconds")
        logger.info(f"Final Accuracy: {metrics['accuracy'][-1]:.4f}")
        logger.info(f"Model Size: {model_size_mb:.2f} MB")
        logger.info(f"Total FLOPs: {total_flops:.2e}")
        
        if not use_dense_baseline:
            logger.info(f"Average Sparsity: {np.mean(metrics['sparsity']):.4f}")
            logger.info(f"Average Compression Ratio: {np.mean(metrics['compression_ratio']):.4f}")
            logger.info(f"Average Effective Compute Ratio: {np.mean(metrics['effective_compute_ratio']):.4f}")
        
        logger.info(f"Total Communication Cost: {metrics['communication'][-1]/1024:.2f} KB")
        logger.info(f"Total Memory Used: {metrics['memory'][-1]/1024/1024:.2f} MB")
        
        if not use_dense_baseline and 'memory_saved' in metrics:
            logger.info(f"Total Memory Saved: {metrics['memory_saved'][-1]/1024/1024:.2f} MB")
        
        # System metrics
        logger.info(f"Average CPU Usage: {monitoring_summary.get('avg_cpu_percent', 'N/A'):.2f}%")
        logger.info(f"Max CPU Usage: {monitoring_summary.get('max_cpu_percent', 'N/A'):.2f}%")
        logger.info(f"Average Memory Usage: {monitoring_summary.get('avg_memory_mb', 'N/A'):.2f} MB")
        logger.info(f"Max Memory Usage: {monitoring_summary.get('max_memory_mb', 'N/A'):.2f} MB")
        
        if monitoring_summary.get('avg_temperature') is not None:
            logger.info(f"Average Temperature: {monitoring_summary.get('avg_temperature'):.2f}°C")
            logger.info(f"Max Temperature: {monitoring_summary.get('max_temperature'):.2f}°C")
        
        # Clean up to free memory
        del model
        gc.collect()
        
        # Combine metrics and monitoring data
        result = {
            'model_name': model_name,
            'training_time': training_time,
            'final_accuracy': metrics['accuracy'][-1],
            'model_size_mb': model_size_mb,
            'total_flops': total_flops,
            'communication_kb': metrics['communication'][-1]/1024,
            'memory_mb': metrics['memory'][-1]/1024/1024,
            'system_metrics': monitoring_summary
        }
        
        if not use_dense_baseline:
            result.update({
                'avg_sparsity': np.mean(metrics['sparsity']),
                'avg_compression_ratio': np.mean(metrics['compression_ratio']),
                'avg_effective_compute_ratio': np.mean(metrics['effective_compute_ratio']),
            })
            
            if 'memory_saved' in metrics:
                result['memory_saved_mb'] = metrics['memory_saved'][-1]/1024/1024
        
        return result
        
    except Exception as e:
        # Stop monitoring in case of error
        monitor.stop()
        logger.error(f"Error during {model_name} benchmarking: {str(e)}")
        raise

def run_benchmarks(rounds=20, batch_size=16, alpha=0.5, num_clients=5):
    """Run benchmarks for both FedTinyProp and Dense models"""
    results = {}
    
    try:
        # Load dataset
        trainset, testset = load_cifar10()
        
        # Partition data
        partitions = partition_data(trainset, alpha=alpha, num_clients=num_clients)
        
        # First run FedTinyProp
        logger.info("Starting FedTinyProp benchmark...")
        fedtinyprop_results = benchmark_model(
            partitions, 
            testset, 
            use_dense_baseline=False,
            rounds=rounds,
            batch_size=batch_size
        )
        results['fedtinyprop'] = fedtinyprop_results
        
        # Then run Dense baseline
        logger.info("Starting Dense model benchmark...")
        dense_results = benchmark_model(
            partitions, 
            testset, 
            use_dense_baseline=True,
            rounds=rounds,
            batch_size=batch_size
        )
        results['dense'] = dense_results
        
        # Compare results
        logger.info("\n===== BENCHMARK COMPARISON =====")
        logger.info(f"FedTinyProp Training Time: {fedtinyprop_results['training_time']:.2f} seconds")
        logger.info(f"Dense Model Training Time: {dense_results['training_time']:.2f} seconds")
        logger.info(f"Time Speedup: {dense_results['training_time']/fedtinyprop_results['training_time']:.2f}x")
        
        logger.info(f"\nFedTinyProp Final Accuracy: {fedtinyprop_results['final_accuracy']:.4f}")
        logger.info(f"Dense Model Final Accuracy: {dense_results['final_accuracy']:.4f}")
        
        logger.info(f"\nFedTinyProp Model Size: {fedtinyprop_results['model_size_mb']:.2f} MB")
        logger.info(f"Dense Model Size: {dense_results['model_size_mb']:.2f} MB")
        logger.info(f"Size Reduction: {dense_results['model_size_mb']/fedtinyprop_results['model_size_mb']:.2f}x")
        
        logger.info(f"\nFedTinyProp Memory Usage: {fedtinyprop_results['memory_mb']:.2f} MB")
        logger.info(f"Dense Model Memory Usage: {dense_results['memory_mb']:.2f} MB")
        logger.info(f"Memory Reduction: {dense_results['memory_mb']/fedtinyprop_results['memory_mb']:.2f}x")
        
        logger.info(f"\nFedTinyProp Communication: {fedtinyprop_results['communication_kb']:.2f} KB")
        logger.info(f"Dense Model Communication: {dense_results['communication_kb']:.2f} KB")
        logger.info(f"Communication Reduction: {dense_results['communication_kb']/fedtinyprop_results['communication_kb']:.2f}x")
        
        logger.info(f"\nFedTinyProp Total FLOPs: {fedtinyprop_results['total_flops']:.2e}")
        logger.info(f"Dense Model Total FLOPs: {dense_results['total_flops']:.2e}")
        logger.info(f"FLOPs Reduction: {dense_results['total_flops']/fedtinyprop_results['total_flops']:.2f}x")
        
        # Save comparison results
        comparison = {
            'time_speedup': dense_results['training_time']/fedtinyprop_results['training_time'],
            'accuracy_fedtinyprop': fedtinyprop_results['final_accuracy'],
            'accuracy_dense': dense_results['final_accuracy'],
            'size_reduction': dense_results['model_size_mb']/fedtinyprop_results['model_size_mb'],
            'memory_reduction': dense_results['memory_mb']/fedtinyprop_results['memory_mb'],
            'communication_reduction': dense_results['communication_kb']/fedtinyprop_results['communication_kb'],
            'flops_reduction': dense_results['total_flops']/fedtinyprop_results['total_flops'],
        }
        
        results['comparison'] = comparison
        
        # Save all results to JSON
        with open('../results/rpi_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Benchmark results saved to results/rpi_benchmark_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        return None

def plot_results(results):
    """Plot benchmark results"""
    if not results:
        logger.error("No results to plot")
        return
    
    try:
        # Create results directory if it doesn't exist
        os.makedirs('../results/plots', exist_ok=True)
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        plt.bar(['FedTinyProp', 'Dense'], 
                [results['fedtinyprop']['final_accuracy'], results['dense']['final_accuracy']])
        plt.title('Final Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/plots/accuracy_comparison.png')
        
        # Plot training time comparison
        plt.figure(figsize=(10, 6))
        plt.bar(['FedTinyProp', 'Dense'], 
                [results['fedtinyprop']['training_time'], results['dense']['training_time']])
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/plots/time_comparison.png')
        
        # Plot memory usage comparison
        plt.figure(figsize=(10, 6))
        plt.bar(['FedTinyProp', 'Dense'], 
                [results['fedtinyprop']['memory_mb'], results['dense']['memory_mb']])
        plt.title('Memory Usage Comparison')
        plt.ylabel('Memory (MB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/plots/memory_comparison.png')
        
        # Plot communication cost comparison
        plt.figure(figsize=(10, 6))
        plt.bar(['FedTinyProp', 'Dense'], 
                [results['fedtinyprop']['communication_kb'], results['dense']['communication_kb']])
        plt.title('Communication Cost Comparison')
        plt.ylabel('Communication (KB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/plots/communication_comparison.png')
        
        # Plot FLOPs comparison
        plt.figure(figsize=(10, 6))
        plt.bar(['FedTinyProp', 'Dense'], 
                [results['fedtinyprop']['total_flops'], results['dense']['total_flops']])
        plt.title('FLOPs Comparison')
        plt.ylabel('FLOPs')
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/plots/flops_comparison.png')
        
        # Plot reduction factors
        plt.figure(figsize=(12, 8))
        metrics = ['time_speedup', 'size_reduction', 'memory_reduction', 
                  'communication_reduction', 'flops_reduction']
        values = [results['comparison'][m] for m in metrics]
        labels = ['Time', 'Model Size', 'Memory', 'Communication', 'FLOPs']
        
        plt.bar(labels, values)
        plt.title('Reduction Factors (Dense / FedTinyProp)')
        plt.ylabel('Reduction Factor')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('../results/plots/reduction_factors.png')
        
        logger.info("Benchmark plots saved to results/plots/")
        
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FedTinyProp Raspberry Pi Benchmarking')
    parser.add_argument('--rounds', type=int, default=20, help='Number of training rounds')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha parameter')
    parser.add_argument('--num-clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--plot-only', action='store_true', help='Only plot existing results')
    
    args = parser.parse_args()
    
    if args.plot_only:
        try:
            with open('../results/rpi_benchmark_results.json', 'r') as f:
                results = json.load(f)
            plot_results(results)
        except Exception as e:
            logger.error(f"Error loading results for plotting: {str(e)}")
    else:
        # Run benchmarks
        results = run_benchmarks(
            rounds=args.rounds,
            batch_size=args.batch_size,
            alpha=args.alpha,
            num_clients=args.num_clients
        )
        
        # Plot results
        if results:
            plot_results(results)
