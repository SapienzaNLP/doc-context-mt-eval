"""LoRA configuration and application."""

import logging
from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

from config_finetuning import LoRAConfig

logger = logging.getLogger(__name__)


def apply_lora(model, lora_config: LoRAConfig):
    """
    Apply LoRA adapter to model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        Model with LoRA applied
    """
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        target_modules=lora_config.target_modules
    )
    
    lora_model = get_peft_model(model, peft_config)

    # Required when gradient_checkpointing=True: without this, gradients don't
    # flow back to LoRA weights because the first layer's input has no grad_fn.
    lora_model.enable_input_require_grads()

    logger.info("LoRA configuration applied successfully")
    logger.info(f"Rank (r): {lora_config.r}")
    logger.info(f"Alpha: {lora_config.lora_alpha}")
    logger.info(f"Dropout: {lora_config.lora_dropout}")
    logger.info(f"Target modules: {lora_config.target_modules}")
    
    return lora_model