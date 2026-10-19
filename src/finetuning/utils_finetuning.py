"""Utility functions for finetuning."""

import logging

logger = logging.getLogger(__name__)


# Language mapping
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


def setup_cuda():
    """Setup CUDA settings."""
    import torch
    
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")


def get_response_template(model_name: str) -> str:
    """
    Return the chat-template token marking the start of the assistant's turn,
    used by `DataCollatorForCompletionOnlyLM` to mask the loss up to that point.

    Each model family uses different turn-marker tokens, so this must match
    the model's own chat template exactly or the collator silently fails to
    find it and masks the entire example.
    """
    name_lower = model_name.lower()
    if "qwen3" in name_lower:
        return "<|im_start|>assistant"  # ChatML
    raise ValueError(
        f"No known response template for model '{model_name}'. "
        "Add a case in get_response_template() for this model family."
    )