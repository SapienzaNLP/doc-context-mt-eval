from typing import List, Dict, Tuple
import re
import json
from json_repair import repair_json
from dataclasses import dataclass


@dataclass
class SpanMatchingStats:
    total: int = 0
    unique_match: int = 0
    ambiguous_no_context: int = 0
    ambiguous_context_not_found: int = 0
    ambiguous_context_not_unique: int = 0
    ambiguous_resolved: int = 0
    ambiguous_fallback: int = 0


def fix_unescaped_quotes(text):
    return re.sub(r'(?<!\\)"(?=.*?:)', r'\"', text)


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        repaired = repair_json(text)
        return json.loads(repaired)
    except Exception:
        pass

    # Extract the last balanced {...} block and retry
    stack = []
    json_blocks = []
    start = None

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start = i
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    json_blocks.append(text[start:i + 1])
                    start = None

    for block in reversed(json_blocks):
        try:
            return json.loads(block)
        except Exception:
            pass
        try:
            repaired = repair_json(block)
            return json.loads(repaired)
        except Exception:
            pass
        try:
            fixed = fix_unescaped_quotes(block)
            return json.loads(fixed)
        except Exception:
            continue

    return None


def normalize_text(text):
    """
    Normalize text without changing character count: lowercasing and
    standardizing quotes, dashes, and spaces.
    """
    text = text.lower()

    TRANSLATION_TABLE = str.maketrans({
        # Quotes
        "’": '"', "‘": '"', "`": '"',
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "'": '"',

        # Dashes
        "–": "-", "—": "-", "−": "-", "‐": "-",

        # Spaces
        " ": " ", " ": " ", " ": " ",
    })

    return text.translate(TRANSLATION_TABLE)


def find_error_positions_with_severity(
    text: str, errors: List[Dict[str, str]], stats: SpanMatchingStats | None = None
) -> List[Tuple[int, int, str]]:
    if stats is None:
        stats = SpanMatchingStats()

    error_positions = []

    for error in errors:
        span = normalize_text(error.get("span") or error.get("minimal_span"))
        if span == '':
            continue

        stats.total += 1
        severity = error['severity']
        positions = [(m.start(), m.end()) for m in re.finditer(re.escape(span), text)]

        if len(positions) == 0:
            continue
        elif len(positions) == 1:
            stats.unique_match += 1
            start_idx, end_idx = positions[0]
        else:
            context = error.get("extended_span") or error.get("span_with_context")
            if not context:
                stats.ambiguous_no_context += 1
                start_idx, end_idx = positions[0]
            else:
                context = normalize_text(context)
                context_positions = [m.start() for m in re.finditer(re.escape(context), text)]

                if len(context_positions) == 0:
                    stats.ambiguous_context_not_found += 1
                    start_idx, end_idx = positions[0]
                else:
                    if len(context_positions) > 1:
                        stats.ambiguous_context_not_unique += 1

                    context_start = context_positions[0]
                    context_end = context_start + len(context)
                    match = next(
                        ((s, e) for s, e in positions if context_start <= s and e <= context_end),
                        None
                    )
                    if match:
                        stats.ambiguous_resolved += 1
                        start_idx, end_idx = match
                    else:
                        stats.ambiguous_fallback += 1
                        start_idx, end_idx = positions[0]

        error_positions.append((start_idx, end_idx, severity))

    return error_positions
