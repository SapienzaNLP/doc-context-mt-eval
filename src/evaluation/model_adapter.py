"""Model adapter providing a unified interface for Qwen3 and Gemma 4 models."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """
    Static profile describing how a model family behaves.
    All fields are derived once at init time from the model name.
    """
    family: str                  # "qwen3" | "gemma4" | "unknown"
    supports_system_role: bool
    thinking_end_token: str      # token that closes the thinking block
    thinking_start_token: str    # token that opens the thinking block (for detection)
    # Recommended sampling params for thinking mode
    thinking_temperature: float
    thinking_top_p: float
    thinking_top_k: int
    thinking_min_p: float
    # Max model context length (tokens)
    max_model_len: int


def get_model_profile(model_name: str) -> ModelProfile:
    """
    Derive the ModelProfile from a model name string.

    Supports:
      - Qwen3 variants  (e.g. "Qwen/Qwen3-8B", "Qwen/Qwen3-32B-Instruct")
      - Gemma 4 variants (e.g. "google/gemma-4-E4B-it", "google/gemma-4-E2B-it")

    Falls back to a conservative "unknown" profile for anything else.
    """
    name_lower = model_name.lower()

    # Qwen3
    if "qwen3" in name_lower:
        return ModelProfile(
            family="qwen3",
            supports_system_role=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            thinking_temperature=0.6,
            thinking_top_p=0.95,
            thinking_top_k=20,
            thinking_min_p=0.0,
            max_model_len=32768,
        )

    # Gemma 4
    if "gemma-4" in name_lower or "gemma4" in name_lower:
        is_small = any(tag in name_lower for tag in ["e2b", "e4b"])
        return ModelProfile(
            family="gemma4",
            supports_system_role=True,
            thinking_start_token="<|channel>thought",
            thinking_end_token="<channel|>",
            thinking_temperature=1.0,
            thinking_top_p=0.95,
            thinking_top_k=64,
            thinking_min_p=0.0,
            max_model_len=131072 if is_small else 262144,
        )

    # Unknown / generic fallback
    logger.warning(
        "Model '%s' is not explicitly supported. "
        "Falling back to generic profile — results may be suboptimal.",
        model_name,
    )
    return ModelProfile(
        family="unknown",
        supports_system_role=False,
        thinking_start_token="<think>",
        thinking_end_token="</think>",
        thinking_temperature=0.6,
        thinking_top_p=0.95,
        thinking_top_k=20,
        thinking_min_p=0.0,
        max_model_len=32768,
    )


# Gemma 4 JSON parsing helpers

def _fix_truncated_json(text: str) -> str:
    """Close a truncated JSON string by appending missing brackets."""
    text = re.sub(r",\s*$", "", text.strip())
    text += "]" * (text.count("[") - text.count("]"))
    text += "}" * (text.count("{") - text.count("}"))
    return text


def _try_load_json(text: str) -> tuple[bool, str]:
    """
    Try to parse text as JSON, then attempt repair if it fails.

    Returns:
        (is_valid, final_text) — final_text is the repaired version if repair
        succeeded, otherwise the original text unchanged.
    """
    try:
        json.loads(text)
        return True, text
    except (json.JSONDecodeError, ValueError):
        pass

    fixed = _fix_truncated_json(text)
    try:
        json.loads(fixed)
        logger.debug("Repaired truncated JSON.")
        return True, fixed
    except (json.JSONDecodeError, ValueError):
        return False, text


def _find_last_balanced_json(text: str) -> tuple[int, int] | None:
    """
    Return (start, end) of the last top-level balanced {...} block in text.
    Scans right-to-left so nested JSON is handled correctly.
    Note: does not account for { } inside quoted strings, which is
    acceptable for typical model outputs.
    """
    depth = 0
    end = -1
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        if c == '}':
            if depth == 0:
                end = i
            depth += 1
        elif c == '{':
            depth -= 1
            if depth == 0 and end != -1:
                return i, end + 1
    return None


def _extract_gemma4_json(content: str) -> tuple[str, str]:
    """
    Extract (thinking, json_response) from Gemma 4 output after the
    thinking block has already been stripped.

    Strategy (in order):
      1. Last ```json … ``` fenced block (greedy inner match for nested JSON).
      2. Last balanced { … } block (stack-based, handles nesting correctly).
      3. First truncated { … to end-of-string, repaired with _fix_truncated_json.
      4. Return (content, "") — caller will log a parse error.
    """
    # 1. Fenced block — greedy inner match handles nested JSON objects
    fenced = list(re.finditer(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL))
    if fenced:
        last = fenced[-1]
        return content[:last.start()].strip(), last.group(1).strip()

    # 2. Last balanced JSON object (stack-based, not regex)
    span = _find_last_balanced_json(content)
    if span:
        start, end = span
        return content[:start].strip(), content[start:end].strip()

    # 3. Truncated JSON
    truncated = re.search(r"(\{.*)", content, flags=re.DOTALL)
    if truncated:
        return content[:truncated.start()].strip(), _fix_truncated_json(truncated.group(1))

    # 4. No JSON found
    return content.strip(), ""


def _parse_gemma4_response(generated_text: str) -> Tuple[str, str]:
    """
    Full parser for Gemma 4 raw output.

    Gemma 4 uses a non-standard thinking format:
        thought\\n<thinking content><channel|><json response>

    After vLLM strips the outer special tokens the text arriving here may:
      - Start with "thought\\n" followed by thinking content.
      - Contain the JSON response in a fenced block or as a bare object.
      - Be truncated mid-JSON (model hit max_tokens).

    Returns:
        (thinking_content, final_response) — both stripped.
        final_response is the raw JSON string (not yet parsed).
    """
    if not generated_text:
        return "", ""

    # Strip the leading "thought\n" prefix emitted by Gemma 4
    content = (
        generated_text[len("thought\n"):]
        if generated_text.startswith("thought\n")
        else generated_text
    )

    thinking, response = _extract_gemma4_json(content)

    if not response:
        logger.warning("Gemma 4: no JSON found in output. Raw text (first 200 chars): %r", content[:200])

    is_valid, response = _try_load_json(response)
    if not is_valid:
        logger.warning("Gemma 4: JSON parse error. Response (first 200 chars): %r", response[:200])

    return thinking, response


# Unified response parser

def parse_model_response(
    generated_text: str,
    enable_thinking: bool,
    profile: ModelProfile,
) -> Tuple[str, str]:
    """
    Split raw model output into (thinking_content, final_response).

    Gemma 4 uses a completely different output format from Qwen3, so it gets
    its own dedicated parser that handles fenced/bare/truncated JSON and the
    non-standard "thought\\n" prefix — no post-hoc reparse script needed.

    For all other families the original token-split logic is used.

    Args:
        generated_text: Raw text produced by the model (already stripped of
                        prompt / special tokens by vLLM).
        enable_thinking: Whether thinking mode was active during generation.
        profile:        ModelProfile for the model that produced the output.

    Returns:
        Tuple of (thinking_content, final_content). Both strings are stripped.
        If thinking is disabled or no thinking block is found, thinking_content
        is an empty string.
    """
    if not enable_thinking:
        return "", generated_text.strip()

    # Gemma 4: dedicated parser
    if profile.family == "gemma4":
        return _parse_gemma4_response(generated_text)

    # Qwen3 / unknown: token-split logic
    end_token = profile.thinking_end_token

    if end_token in generated_text:
        thinking_raw, final_raw = generated_text.split(end_token, 1)
        return thinking_raw.strip(), final_raw.strip()

    fallback_token = "</think>" if end_token != "</think>" else "<channel|>"
    if fallback_token in generated_text:
        logger.warning(
            "Expected thinking end token '%s' not found; "
            "falling back to '%s'. Check that the correct model profile is used.",
            end_token,
            fallback_token,
        )
        thinking_raw, final_raw = generated_text.split(fallback_token, 1)
        return thinking_raw.strip(), final_raw.strip()

    logger.debug("No thinking end token found in generated text; returning empty thinking.")
    return "", generated_text.strip()


# Chat template

def apply_chat_template(
    tokenizer,
    messages: list,
    enable_thinking: bool,
    profile: ModelProfile,
) -> str:
    """
    Apply the correct chat template for the given model family.

    Handles:
      - System role injection for models that don't support it natively
        (inlines the system message as a prefix of the first user turn).
      - enable_thinking flag routing (passed only when the tokenizer supports it).
      - Safe fallback if apply_chat_template raises TypeError.

    Args:
        tokenizer:       HuggingFace tokenizer (or AutoProcessor).
        messages:        Standard OpenAI-style message list.
        enable_thinking: Whether to activate thinking mode.
        profile:         ModelProfile for the active model.

    Returns:
        Formatted prompt string ready to be fed to vLLM.
    """
    working_messages = list(messages)

    # System role handling
    if not profile.supports_system_role:
        system_content = next(
            (m["content"] for m in working_messages if m["role"] == "system"),
            None,
        )
        working_messages = [m for m in working_messages if m["role"] != "system"]
        if system_content and working_messages:
            first = working_messages[0]
            working_messages[0] = {
                "role": first["role"],
                "content": f"{system_content}\n\n{first['content']}",
            }

    # Template application
    base_kwargs = dict(tokenize=False, add_generation_prompt=True)

    try:
        return tokenizer.apply_chat_template(
            working_messages,
            **base_kwargs,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        pass

    logger.debug(
        "Tokenizer for '%s' does not support enable_thinking; "
        "applying template without it.",
        profile.family,
    )
    return tokenizer.apply_chat_template(working_messages, **base_kwargs)