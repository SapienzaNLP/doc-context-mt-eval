"""Prompt templates for MT evaluation — model-agnostic via model_adapter."""

from typing import Any
from prompt_templates import (
    SYSTEM_MESSAGE,
    BASE_REF_LESS_TEMPLATE,
    BASE_REF_BASED_TEMPLATE,
    REPEAT_REF_LESS_TEMPLATE,
    DOC_REF_LESS_TEMPLATE,
    DOC_REF_BASED_TEMPLATE,
)
from model_adapter import ModelProfile, apply_chat_template


class PromptGenerator:
    """Generate prompts for MT evaluation."""

    @staticmethod
    def generate_paragraph_prompt(
        source_language: str,
        target_language: str,
        source_para: str,
        target_para: str,
        reference_para: str,
        is_reference_less: bool,
        enable_thinking: bool,
        tokenizer: Any,
        profile: ModelProfile,
    ) -> str:
        """Generate paragraph-level evaluation prompt."""
        if is_reference_less:
            user_content = BASE_REF_LESS_TEMPLATE.format(
                source_language=source_language,
                target_language=target_language,
                source_para=source_para,
                target_para=target_para,
            )
        else:
            user_content = BASE_REF_BASED_TEMPLATE.format(
                source_language=source_language,
                target_language=target_language,
                source_para=source_para,
                reference_para=reference_para,
                target_para=target_para,
            )

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user",   "content": user_content},
        ]
        return apply_chat_template(tokenizer, messages, enable_thinking, profile)

    @staticmethod
    def generate_repeat_prompt(
        source_language: str,
        target_language: str,
        source_para: str,
        target_para: str,
        enable_thinking: bool,
        tokenizer: Any,
        profile: ModelProfile,
    ) -> str:
        """Generate a reference-less prompt with the segment duplicated in place
        of document context (repetition control — isolates any benefit of
        real document context from a mere "extra text" / prompt-length effect)."""
        user_content = REPEAT_REF_LESS_TEMPLATE.format(
            source_language=source_language,
            target_language=target_language,
            source_para=source_para,
            target_para=target_para,
        )

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user",   "content": user_content},
        ]
        return apply_chat_template(tokenizer, messages, enable_thinking, profile)

    @staticmethod
    def generate_document_prompt(
        source_language: str,
        target_language: str,
        source_doc: str,
        target_doc: str,
        source_para: str,
        target_para: str,
        reference_para: str,
        is_reference_less: bool,
        enable_thinking: bool,
        tokenizer: Any,
        profile: ModelProfile,
    ) -> str:
        """Generate document-level evaluation prompt."""
        if is_reference_less:
            user_content = DOC_REF_LESS_TEMPLATE.format(
                source_language=source_language,
                target_language=target_language,
                source_doc=source_doc,
                target_doc=target_doc,
                source_para=source_para,
                target_para=target_para,
            )
        else:
            user_content = DOC_REF_BASED_TEMPLATE.format(
                source_language=source_language,
                target_language=target_language,
                source_doc=source_doc,
                target_doc=target_doc,
                source_para=source_para,
                reference_para=reference_para,
                target_para=target_para,
            )

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user",   "content": user_content},
        ]
        return apply_chat_template(tokenizer, messages, enable_thinking, profile)