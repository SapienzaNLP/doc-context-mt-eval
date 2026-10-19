"""Utility functions for evaluation."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load preprocessed dataset from JSONL file.
    Returns a list of dataset entries.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping invalid JSON at line {line_num}: {e}"
                )
    
    logger.info(f"Loaded {len(data)} entries from {dataset_path}")
    return data


def setup_cuda():
    """Setup CUDA and log device information."""
    import torch
    
    torch.cuda.empty_cache()
    
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    
    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"CUDA version: {cuda_version}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA devices: {device_count}")
        for i in range(device_count):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    return cuda_available