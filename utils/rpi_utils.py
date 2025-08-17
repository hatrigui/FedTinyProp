"""
Raspberry Pi specific utility functions for measuring hardware metrics.
"""
import os
import subprocess
import psutil
from typing import Dict, Any, Optional

def get_raspberry_pi_memory_usage() -> Dict[str, float]:
    """
    Get memory usage statistics for Raspberry Pi.
    
    Returns:
        Dict with memory usage information in MB:
            - total_memory_mb: Total physical memory
            - available_memory_mb: Available memory
            - used_memory_mb: Used memory
            - sram_used_mb: SRAM usage (approximated from used memory)
    """
    try:
        # Get memory information using psutil
        mem = psutil.virtual_memory()
        
        # Convert bytes to MB
        total_memory_mb = mem.total / (1024 * 1024)
        available_memory_mb = mem.available / (1024 * 1024)
        used_memory_mb = mem.used / (1024 * 1024)
        
        # For Raspberry Pi, we'll use the used memory as an approximation of SRAM usage
        # This is a simplification, but provides a reasonable metric for tracking
        sram_used_mb = used_memory_mb
        
        return {
            'total_memory_mb': total_memory_mb,
            'available_memory_mb': available_memory_mb,
            'used_memory_mb': used_memory_mb,
            'sram_used_mb': sram_used_mb
        }
    except Exception as e:
        print(f"Error getting Raspberry Pi memory usage: {e}")
        return {
            'total_memory_mb': 0,
            'available_memory_mb': 0,
            'used_memory_mb': 0,
            'sram_used_mb': 0
        }

def get_raspberry_pi_temperature() -> float:
    """
    Get the CPU temperature of the Raspberry Pi.
    
    Returns:
        CPU temperature in Celsius, or 0 if unable to retrieve.
    """
    try:
        # Try to read temperature from thermal zone
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                return temp
        
        # Alternative method using vcgencmd
        try:
            output = subprocess.check_output(['vcgencmd', 'measure_temp'], universal_newlines=True)
            temp_str = output.split('=')[1].split("'")[0]
            return float(temp_str)
        except (subprocess.SubprocessError, IndexError, ValueError):
            pass
            
        return 0.0
    except Exception as e:
        print(f"Error getting Raspberry Pi temperature: {e}")
        return 0.0

def get_raspberry_pi_cpu_usage() -> float:
    """
    Get the CPU usage percentage of the Raspberry Pi.
    
    Returns:
        CPU usage as a percentage (0-100), or 0 if unable to retrieve.
    """
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception as e:
        print(f"Error getting Raspberry Pi CPU usage: {e}")
        return 0.0

def check_throttling() -> Dict[str, bool]:
    """
    Check if the Raspberry Pi is being throttled due to temperature or power issues.
    
    Returns:
        Dict with throttling status information:
            - throttled: True if any throttling is occurring
            - temperature_limit: True if temperature limit is active
            - voltage_limit: True if voltage limit is active
    """
    try:
        # Use vcgencmd to get throttling information
        output = subprocess.check_output(['vcgencmd', 'get_throttled'], universal_newlines=True)
        throttled_hex = output.split('=')[1].strip()
        throttled_int = int(throttled_hex, 16)
        
        # Decode the throttling bits
        return {
            'throttled': throttled_int > 0,
            'temperature_limit': bool(throttled_int & (1 << 0)),
            'voltage_limit': bool(throttled_int & (1 << 1)),
            'currently_throttled': bool(throttled_int & (1 << 16)),
            'soft_temperature_limit': bool(throttled_int & (1 << 17)),
        }
    except Exception as e:
        print(f"Error checking Raspberry Pi throttling: {e}")
        return {
            'throttled': False,
            'temperature_limit': False,
            'voltage_limit': False,
            'currently_throttled': False,
            'soft_temperature_limit': False,
        }
