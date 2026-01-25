"""
Memory Monitoring Utilities

Simple tools for tracking memory usage in parallel experiments.
"""

import psutil
import gc


def get_memory_info():
    """Get current memory statistics."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'used_gb': mem.used / 1e9,
        'available_gb': mem.available / 1e9,
        'percent': mem.percent,
        'swap_gb': swap.used / 1e9,
    }


def print_memory(prefix=""):
    """Print memory status."""
    info = get_memory_info()
    print(f"{prefix}Memory: {info['percent']:.1f}% used, "
          f"{info['available_gb']:.1f}GB free, "
          f"Swap: {info['swap_gb']:.1f}GB")
    
    return info


def check_memory_safe(threshold=85.0, swap_threshold=1.0):
    """
    Check if memory is safe for continued operation.
    
    Returns (is_safe, message)
    """
    info = get_memory_info()
    
    if info['swap_gb'] > swap_threshold:
        return False, f"⚠️ Swap active: {info['swap_gb']:.1f}GB"
    
    if info['percent'] > threshold:
        return False, f"⚠️ Memory high: {info['percent']:.1f}%"
    
    return True, "✓ Memory OK"


def memory_cleanup():
    """Force garbage collection."""
    collected = gc.collect()
    return collected


class MemoryMonitor:
    """Context manager for memory monitoring."""
    
    def __init__(self, label="", warn_threshold=85.0):
        self.label = label
        self.warn_threshold = warn_threshold
        self.start_info = None
    
    def __enter__(self):
        self.start_info = get_memory_info()
        if self.label:
            print(f"[{self.label}] Start - Memory: {self.start_info['percent']:.1f}%")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_info = get_memory_info()
        delta = end_info['used_gb'] - self.start_info['used_gb']
        
        if self.label:
            print(f"[{self.label}] End - Memory: {end_info['percent']:.1f}% "
                  f"(Δ{delta:+.1f}GB)")
        
        if end_info['percent'] > self.warn_threshold:
            print(f"  ⚠️ Memory warning: {end_info['percent']:.1f}%")