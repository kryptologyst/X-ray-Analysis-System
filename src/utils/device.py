"""Utility functions for device management and deterministic behavior."""

import random
import numpy as np
import torch
import os
from typing import Optional, Union


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device with fallback.
    
    Args:
        device: Preferred device ('cuda', 'mps', 'cpu', 'auto')
        
    Returns:
        torch.device: The selected device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "device_count": 0,
        "current_device": None
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory"] = torch.cuda.get_device_properties(0).total_memory
    
    return info


def move_to_device(data: Union[torch.Tensor, dict, list], device: torch.device) -> Union[torch.Tensor, dict, list]:
    """Move data to specified device.
    
    Args:
        data: Data to move (tensor, dict, or list)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def get_memory_usage(device: torch.device) -> dict:
    """Get memory usage information for device.
    
    Args:
        device: Device to check
        
    Returns:
        dict: Memory usage information
    """
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device)
        }
    else:
        return {"message": f"Memory tracking not available for {device.type}"}


def clear_memory(device: torch.device) -> None:
    """Clear device memory cache.
    
    Args:
        device: Device to clear
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def enable_deterministic() -> None:
    """Enable deterministic behavior across all operations."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_deterministic() -> None:
    """Disable deterministic behavior for better performance."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
