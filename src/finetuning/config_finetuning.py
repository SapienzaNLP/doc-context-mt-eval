"""Configuration for model finetuning."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import numpy as np


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
class PromptTemplateConfig:
    """Configuration for which prompt templates to generate."""
    include_document_context: bool = True
    is_reference_less: bool = False
    use_all_templates: bool = False

    def get_active_templates(self) -> List[tuple]:
        """
        Get list of active template configurations.

        Returns:
            List of (include_document, is_reference_less) tuples
        """
        if self.use_all_templates:
            return [
                (False, True),   # Para ref-less
                (False, False),  # Para ref-based
                (True, True),    # Doc ref-less
                (True, False),   # Doc ref-based
            ]
        else:
            return [(self.include_document_context, self.is_reference_less)]


@dataclass
class ModelConfig:
    """Configuration for model and tokenizer."""
    model_name: str
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"

    # Training hyperparameters (paper Table 9)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_length: int = 8192
    warmup_steps: int = 100

    # Optimizer settings
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "linear"

    # Training settings
    gradient_checkpointing: bool = True
    bf16: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter (paper Table 9)."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class DataConfig:
    """Configuration for data preparation."""
    data_paths: List[Path]
    exclude_domains: List[str] = None
    language_pairs: List[str] = None
    test_size: float = 0.1

    def __post_init__(self):
        if not (0 < self.test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")


@dataclass
class FinetuningConfig:
    """Main configuration for finetuning."""
    model_config: ModelConfig
    data_config: DataConfig
    lora_config: LoRAConfig
    seed_config: SeedConfig
    template_config: PromptTemplateConfig

    language_pairs: List[str]

    wandb_project_name: str
    wandb_run_name: str

    output_dir: str
    finetuning_data_dir: str = "src/finetuning/finetuning_data"
    wandb_mode: str = "online"

    def get_output_path(self) -> Path:
        """Get output directory path."""
        return Path(self.output_dir)

    def get_data_save_path(self, split: str) -> Path:
        """Get path to save train/eval dataset."""
        return Path(self.finetuning_data_dir) / f"{split}_dataset.json"

    def save_run_config(self, output_dir: Path, filename: str = "run_config.txt") -> None:
        """
        Serialize all run parameters to a human-readable .txt file.

        Args:
            output_dir: Directory where the file will be saved.
            filename:   Output filename (default: run_config.txt).
        """
        mc = self.model_config
        dc = self.data_config
        lc = self.lora_config
        sc = self.seed_config
        tc = self.template_config

        sep = "=" * 80

        lines = [
            sep,
            "RUN CONFIGURATION",
            sep,
            "",
            "  MODEL",
            f"    model_name                  : {mc.model_name}",
            f"    torch_dtype                 : {mc.torch_dtype}",
            f"    device_map                  : {mc.device_map}",
            "",
            "  TRAINING",
            f"    learning_rate               : {mc.learning_rate}",
            f"    weight_decay                : {mc.weight_decay}",
            f"    num_epochs                  : {mc.num_epochs}",
            f"    train_batch_size            : {mc.train_batch_size}",
            f"    eval_batch_size             : {mc.eval_batch_size}",
            f"    gradient_accumulation_steps : {mc.gradient_accumulation_steps}",
            f"    max_length                  : {mc.max_length}",
            f"    warmup_steps                : {mc.warmup_steps}",
            f"    optimizer                   : {mc.optimizer}",
            f"    lr_scheduler_type           : {mc.lr_scheduler_type}",
            f"    gradient_checkpointing      : {mc.gradient_checkpointing}",
            f"    bf16                        : {mc.bf16}",
            "",
            "  LORA",
            f"    r                           : {lc.r}",
            f"    lora_alpha                  : {lc.lora_alpha}",
            f"    lora_dropout                : {lc.lora_dropout}",
            f"    bias                        : {lc.bias}",
            f"    target_modules              : {', '.join(lc.target_modules)}",
            "",
            "  DATA",
            f"    language_pairs              : {', '.join(dc.language_pairs or [])}",
            f"    test_size                   : {dc.test_size}",
            f"    exclude_domains             : {', '.join(dc.exclude_domains) if dc.exclude_domains else '(none)'}",
            "    data_paths                  :",
        ]
        for p in dc.data_paths:
            lines.append(f"      - {p}")

        lines += [
            "",
            "  TEMPLATE",
            f"    use_all_templates           : {tc.use_all_templates}",
            f"    include_document_context    : {tc.include_document_context}",
            f"    is_reference_less           : {tc.is_reference_less}",
        ]

        active = tc.get_active_templates()
        template_labels = {
            (False, True):  "paragraph-level, reference-less",
            (False, False): "paragraph-level, reference-based",
            (True,  True):  "document-level,  reference-less",
            (True,  False): "document-level,  reference-based",
        }
        lines.append(f"    active template(s)          :")
        for t in active:
            lines.append(f"      - {template_labels[t]}")

        lines += [
            "",
            "  OUTPUT & LOGGING",
            f"    output_dir                  : {self.output_dir}",
            f"    finetuning_data_dir         : {self.finetuning_data_dir}",
            f"    wandb_project_name          : {self.wandb_project_name}",
            f"    wandb_run_name              : {self.wandb_run_name}",
            f"    wandb_mode                  : {self.wandb_mode}",
            "",
            "  REPRODUCIBILITY",
            f"    seed                        : {sc.seed}",
            "",
            sep,
            "",
        ]

        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / filename
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        import logging
        logging.getLogger(__name__).info(f"Run config saved to {config_file}")