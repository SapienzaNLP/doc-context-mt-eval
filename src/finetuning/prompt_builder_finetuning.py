"""Prompt generation for finetuning."""

import json
import logging
from typing import Dict, List, Optional
from span_with_context import SpanContextExtractor, SpanValidator
from prompt_templates_finetuning import (
    SYSTEM_MESSAGE,
    BASE_REF_LESS_TEMPLATE,
    BASE_REF_BASED_TEMPLATE,
    DOC_REF_LESS_TEMPLATE,
    DOC_REF_BASED_TEMPLATE,
)

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build training prompts with gold responses."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def _format_errors(
        self, 
        errors: List[Dict], 
        translation: str,
        source: str,
        target_language: Optional[str] = None,
        source_language: Optional[str] = None
    ) -> str:
        """
        Format errors into JSON response string.
        
        Args:
            errors: List of error dictionaries with 'start', 'end', 'category', 'severity'
            translation: Full translation text to extract spans
            target_language: Target language for CJK detection
            
        Returns:
            JSON string with formatted errors
        """
        if not errors:
            return json.dumps({"errors": []}, ensure_ascii=False) 
        
        formatted_errors = {"errors": []}
        
        for err in errors:
            # Use the new extract_with_source_target method
            span_with_context = SpanContextExtractor.extract_with_source_target(
                error=err,
                source_text=source,
                target_text=translation,
                source_language=source_language,
                target_language=target_language
            )
            
            # Skip if span extraction failed
            if not span_with_context:
                continue

            # Validate indices
            if not (isinstance(err.get('start'), int) and isinstance(err.get('end'), int)):
                logger.warning(f"Invalid error indices: {err}")
                continue
            
            start, end = err['start'], err['end']
            
            # Validate span
            if not SpanValidator.validate(start, end, translation):
                continue
            
            # Extract the minimal error span
            if err['is_source_error']:
                error_span = source[start:end]
            else:
                error_span = translation[start:end]
            
            formatted_errors["errors"].append({
                "minimal_span": error_span,
                "extended_span": span_with_context,
                "type": err['category'],
                "severity": err['severity']
            })
        
        return json.dumps(formatted_errors, ensure_ascii=False)
    
    def build_paragraph_prompt(
        self,
        source_lang: str,
        target_lang: str,
        source_para: str,
        reference_para: str,
        translation: str,
        errors: List[Dict],
        is_reference_less: bool
    ) -> str:
        """Build paragraph-level prompt with gold response."""
        response_str = self._format_errors(errors, translation, source_para, target_lang, source_lang)
        
        # Select template
        if is_reference_less:
            user_content = BASE_REF_LESS_TEMPLATE.format(
                source_language=source_lang,
                target_language=target_lang,
                source_para=source_para,
                target_para=translation
            )
        else:
            user_content = BASE_REF_BASED_TEMPLATE.format(
                source_language=source_lang,
                target_language=target_lang,
                source_para=source_para,
                reference_para=reference_para,
                target_para=translation
            )
        
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response_str}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False, # We already include the response in the prompt, so no need to add a generation prompt
            enable_thinking=False
        )
    
    def build_document_prompt(
        self,
        source_lang: str,
        target_lang: str,
        source_doc: str,
        target_doc: str,
        source_para: str,
        reference_para: str,
        translation: str,
        errors: List[Dict],
        is_reference_less: bool
    ) -> str:
        """Build document-level prompt with gold response."""
        response_str = self._format_errors(errors, translation, source_para, target_lang, source_lang)
        
        # Select template
        if is_reference_less:
            user_content = DOC_REF_LESS_TEMPLATE.format(
                source_language=source_lang,
                target_language=target_lang,
                source_doc=source_doc,
                target_doc=target_doc,
                source_para=source_para,
                target_para=translation
            )
        else:
            user_content = DOC_REF_BASED_TEMPLATE.format(
                source_language=source_lang,
                target_language=target_lang,
                source_doc=source_doc,
                target_doc=target_doc,
                source_para=source_para,
                reference_para=reference_para,
                target_para=translation
            )
        
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response_str}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # We already include the response in the prompt, so no need to add a generation prompt
            enable_thinking=False
        )