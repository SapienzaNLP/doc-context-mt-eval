"""Configuration classes for WMT data preprocessing."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProcessingConfig:
    """Configuration for dataset processing."""
    dataset: str
    language_pairs: List[str]
    enable_context_windows: bool
    n_pre: Optional[int]
    n_post: Optional[int]
    max_context: Optional[int]
    output_file: str