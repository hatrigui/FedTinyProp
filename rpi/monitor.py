#!/usr/bin/env python3
# FedTinyProp Raspberry Pi Monitoring Utility
# This script provides real-time monitoring of system resources during training

import time
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Check if running on Raspberry Pi
is_raspberry_pi = os.path.exists('/sys/class/thermal/thermal_zone0/temp')

class RaspberryPiMonitor:
    """Monitor system resources on Raspberry Pi"""
    
    def __init__(self, interval=1.0, output_file=None):
        self.interval = interval
        self.output_file = output_file or f"rpi_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.metrics = {
            'timestamp': [],
            'cpu_percent': [],
            'memory_used_mb': [],
            'memory_total_mb': [],
            'memory_percent': [],
            'temperature': [],
            'throttled': []
        }
        
    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        try:
            with open('/proc/stat', 'r') as f:
                cpu_stats = f.readline().split()
            
            user = float(cpu_stats[1])
            nice = float(cpu_stats[2])
            system = float(cpu_stats[3])
            idle = float(cpu_stats[4])
            iowait = float(cpu_stats[5])
            irq = float(cpu_stats[6])
            softirq = float(cpu_stats[7])
            
            total = user + nice + system + idle + iowait + irq + softirq
            idle_total = idle + iowait
            
            # Store current values for next calculation
            self.prev_total = total
            self.prev_idle = idle_total
            
            # Calculate CPU percentage
            if hasattr(self, 'prev_total') and hasattr(self, 'prev_idle'):
                total_diff = total - self.prev_total
                idle_diff = idle_total - self.prev_idle
                
                if total_diff > 0:
                    cpu_percent = 100.0 * (1.0 - idle_diff / total_diff)
                    return cpu_percent
            
            return 0.0
        except Exception as e:
            print(f"Error getting CPU usage: {str(e)}")
            return 0.0
    
    def get_memory_usage(self):
        """Get memory usage"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.readlines()
            
            mem_total = None
            mem_free = None
            mem_available = None
            
            for line in meminfo:
                if 'MemTotal' in line:
                    mem_total = int(line.split()[1])
                elif 'MemFree' in line:
                    mem_free = int(line.split()[1])
                elif 'MemAvailable' in line:
                    mem_available = int(line.split()[1])
            
            # Convert from KB to MB
            mem_total_mb = mem_total / 1024
            mem_used_mb = (mem_total - mem_available) / 1024
            mem_percent = (mem_used_mb / mem_total_mb) * 100
            
            return {
                'total_mb': mem_total_mb,
                'used_mb': mem_used_mb,
                'percent': mem_percent
            }
        except Exception as e:
            print(f"Error getting memory usage: {str(e)}")
            return {'total_mb': 0, 'used_mb': 0, 'percent': 0}
    
    def get_temperature(self):
        """Get CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except Exception as e:
            print(f"Error getting temperature: {str(e)}")
            return None
    
    def get_throttling_status(self):
        """Get throttling status"""
        try:
            with open('/sys/devices/platform/soc/soc:firmware/get_throttled', 'r') as f:
                throttled = int(f.read().strip(), 0)
            return throttled
        except Exception as e:
            print(f"Error getting throttling status: {str(e)}")
            return 0
    
    def collect_metrics(self):
        """Collect all metrics once"""
        timestamp = time.time()
        cpu_percent = self.get_cpu_usage()
        memory = self.get_memory_usage()
        temperature = self.get_temperature()
        throttled = self.get_throttling_status()
        
        self.metrics['timestamp'].append(timestamp)
        self.metrics['cpu_percent'].append(cpu_percent)
        self.metrics['memory_used_mb'].append(memory['used_mb'])
        self.metrics['memory_total_mb'].append(memory['total_mb'])
        self.metrics['memory_percent'].append(memory['percent'])
        self.metrics['temperature'].append(temperature if temperature is not None else 0)
        self.metrics['throttled'].append(throttled)
        
        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_used_mb': memory['used_mb'],
            'memory_total_mb': memory['total_mb'],
            'memory_percent': memory['percent'],
            'temperature': temperature,
            'throttled': throttled
        }
    
    def start_monitoring(self, duration=None):
        """Start monitoring for specified duration or indefinitely"""
        print(f"Starting Raspberry Pi monitoring (interval: {self.interval}s)")
        print(f"Press Ctrl+C to stop monitoring")
        
        start_time = time.time()
        try:
            while duration is None or (time.time() - start_time) < duration:
                metrics = self.collect_metrics()
                
                # Print current metrics
                print(f"\rCPU: {metrics['cpu_percent']:.1f}% | "
                      f"Memory: {metrics['memory_used_mb']:.1f}/{metrics['memory_total_mb']:.1f} MB "
                      f"({metrics['memory_percent']:.1f}%) | ", end="")
                
                if metrics['temperature'] is not None:
                    print(f"Temp: {metrics['temperature']:.1f}°C | ", end="")
                
                if metrics['throttled'] > 0:
                    print(f"THROTTLING DETECTED: 0x{metrics['throttled']:x} | ", end="")
                
                print(f"Running: {time.time() - start_time:.1f}s", end="")
                
                # Save metrics periodically
                if len(self.metrics['timestamp']) % 10 == 0:
                    self.save_metrics()
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.save_metrics()
            print(f"Metrics saved to {self.output_file}")
    
    def save_metrics(self):
        """Save metrics to CSV file"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.output_file, index=False)
    
    def plot_metrics(self):
        """Plot collected metrics"""
        if not self.metrics['timestamp']:
            print("No metrics to plot")
            return
        
        # Convert timestamps to relative time in seconds
        start_time = self.metrics['timestamp'][0]
        relative_time = [t - start_time for t in self.metrics['timestamp']]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot CPU usage
        axs[0].plot(relative_time, self.metrics['cpu_percent'])
        axs[0].set_title('CPU Usage')
        axs[0].set_ylabel('CPU (%)')
        axs[0].grid(True)
        
        # Plot memory usage
        axs[1].plot(relative_time, self.metrics['memory_used_mb'])
        axs[1].set_title('Memory Usage')
        axs[1].set_ylabel('Memory (MB)')
        axs[1].grid(True)
        
        # Plot temperature if available
        if any(t > 0 for t in self.metrics['temperature']):
            axs[2].plot(relative_time, self.metrics['temperature'])
            axs[2].set_title('CPU Temperature')
            axs[2].set_ylabel('Temperature (°C)')
            axs[2].grid(True)
        
        # Set common x-axis label
        axs[2].set_xlabel('Time (seconds)')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_file.replace('.csv', '.png')
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
        
        # Show plot if running interactively
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi System Monitor')
    parser.add_argument('--interval', type=float, default=1.0, 
                        help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=float, default=None,
                        help='Monitoring duration in seconds (default: indefinite)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for metrics')
    parser.add_argument('--plot', action='store_true',
                        help='Plot metrics after monitoring')
    
    args = parser.parse_args()
    
    if not is_raspberry_pi:
        print("WARNING: This script is designed for Raspberry Pi. Some features may not work.")
    
    monitor = RaspberryPiMonitor(interval=args.interval, output_file=args.output)
    monitor.start_monitoring(duration=args.duration)
    
    if args.plot:
        monitor.plot_metrics()

if __name__ == "__main__":
    main()
