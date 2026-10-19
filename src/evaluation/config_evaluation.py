"""Configuration classes for evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional
import os
import torch
import numpy as np
import random


@dataclass
class SeedConfig:
    """Configuration for reproducibility."""
    seed: int = 42
    
    def apply(self):
        """Apply seed to all random generators."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class ModelConfig:
    """Configuration for model loading and generation."""
    model_name: str
    dtype: torch.dtype = torch.bfloat16
    tensor_parallel_size: int = 2
    max_new_tokens: int = 32768
    
    # Generation parameters for thinking mode
    thinking_temperature: float = 0.6
    thinking_top_p: float = 0.95
    thinking_top_k: int = 20
    thinking_min_p: float = 0.0
    
    # Generation parameters for non-thinking mode (greedy decoding)
    normal_temperature: float = 0.0

    def to_log_dict(self, enable_thinking: bool) -> dict:
        """Return only the relevant sampling parameters for logging."""
        base = {
            "model_name": self.model_name,
            "dtype": str(self.dtype),
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_new_tokens": self.max_new_tokens,
        }
        if enable_thinking:
            base.update({
                "thinking_temperature": self.thinking_temperature,
                "thinking_top_p": self.thinking_top_p,
                "thinking_top_k": self.thinking_top_k,
                "thinking_min_p": self.thinking_min_p,
            })
        else:
            base["normal_temperature"] = self.normal_temperature
        return base


@dataclass
class DataFilterConfig:
    """Configuration for filtering dataset items."""
    exclude_domains: List[str] = field(default_factory=lambda: ['speech'])
    
    def should_include(self, item: dict, std_ref: str) -> bool:
        """Check if an item should be included based on filters."""
        if item.get('para_errors') is None and item.get('gold_errors') is None:
            return False
        
        if item.get('system') == std_ref:
            return False
        
        domain_field = (item.get('domain') or '').lower().strip()
        doc_name = (item.get('doc_name') or '').lower()
        if domain_field:
            if domain_field in self.exclude_domains:
                return False
        elif any(d in doc_name for d in self.exclude_domains):
            return False
        
        return True


@dataclass
class EvaluationConfig:
    """Main configuration for evaluation."""
    # Dataset parameters
    dataset: str
    language_pair: str

    data_path: str = None
    
    # Evaluation mode
    is_reference_less: bool = False
    include_document_context: bool = True
    repeat_segment: bool = False
    enable_thinking: bool = False
    
    # Processing parameters
    batch_size: int = 8
    
    # Output parameters
    output_dir: str = "outputs/metrics_results/"
    
    # WandB parameters
    wandb_project_name: str = "mt-evaluation"
    wandb_run_name: str = "eval_run"
    wandb_mode: str = "online"
    
    # Sub-configurations
    model_config: Optional[ModelConfig] = field(default=None)
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    filter_config: DataFilterConfig = field(default_factory=DataFilterConfig)
    
    @property
    def dataset_path(self) -> str:
        if self.data_path is not None:
            return self.data_path
        return (
            f"data-wmt/preprocessed/mqm/{self.dataset}/"
            f"preprocessed_{self.dataset}_{self.language_pair}.jsonl"
        )

    @property
    def output_path(self) -> str:
        """Get the full output path for results."""
        return os.path.join(
            self.output_dir,
            self.dataset,
            self.language_pair,
            self.model_config.model_name,
            self.wandb_run_name,
            "predicted_score"
        )

    def to_log_dict(self) -> dict:
        """Return the full configuration as a loggable dict."""
        return {
            "dataset": self.dataset,
            "language_pair": self.language_pair,
            "is_reference_less": self.is_reference_less,
            "include_document_context": self.include_document_context,
            "repeat_segment": self.repeat_segment,
            "enable_thinking": self.enable_thinking,
            "batch_size": self.batch_size,
            "model_config": self.model_config.to_log_dict(self.enable_thinking),
            "seed": self.seed_config.seed,
            "exclude_domains": self.filter_config.exclude_domains,
        }


# Language mapping for full names
LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "cs": "Czech",
    "uk": "Ukrainian",
    "is": "Icelandic",
    "hi": "Hindi",
    "ru": "Russian",
    "ko": "Korean",
    "ko_KR": "Korean",
    "zh_CN": "Chinese",
}


def get_language_names(lang_pair: str) -> tuple[str, str]:
    """
    Get full language names from language pair code.
    
    Args:
        lang_pair: Language pair in format "src-tgt" (e.g., "en-de")
        
    Returns:
        Tuple of (source_language, target_language) full names
    """
    src, tgt = lang_pair.split('-')
    return LANGUAGE_MAP.get(src, src), LANGUAGE_MAP.get(tgt, tgt)