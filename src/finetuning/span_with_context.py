"""Span context utilities for error annotation."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SpanContextExtractor:
    """Extract error spans with a fixed context window of 4 tokens on each side."""

    # CJK languages that don't use spaces between words
    CJK_LANGUAGES = {
        'Chinese', 'Japanese', 'Korean',
        'zh', 'ja', 'ko', 'zh-CN', 'zh-TW', 'ja-JP', 'ko-KR'
    }

    CONTEXT_SIZE = 4  # words (standard) or characters (CJK)

    @staticmethod
    def extract(
        span: str,
        start: int,
        end: int,
        full_text: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Extract span with a fixed context window of 4 tokens on each side.

        Args:
            span:      The minimal error span.
            start:     Start position of the span in full_text.
            end:       End position of the span in full_text.
            full_text: Complete text.
            language:  Target language (optional, for CJK detection).

        Returns:
            Span with 4-token context on each side (clipped at text boundaries).
        """
        is_cjk = SpanContextExtractor._is_cjk_language(language)

        if is_cjk:
            return SpanContextExtractor._extract_cjk(start, end, full_text)
        else:
            return SpanContextExtractor._extract_standard(start, end, full_text)

    @staticmethod
    def extract_with_source_target(
        error: dict,
        source_text: str,
        target_text: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> str:
        """
        Extract span with context, handling both source and target errors.

        Args:
            error:           Error dictionary with 'start', 'end', 'is_source_error'.
            source_text:     Source text.
            target_text:     Target/translation text.
            source_language: Source language (for CJK detection).
            target_language: Target language (for CJK detection).

        Returns:
            Span with 4-token context on each side.
        """
        start = error.get('start')
        end = error.get('end')
        is_source_error = error.get('is_source_error', False)

        if not (isinstance(start, int) and isinstance(end, int)):
            logger.warning(f"Invalid error indices: {error}")
            return ""

        if is_source_error:
            full_text = source_text
            language = source_language
            if not SpanValidator.validate(start, end, source_text):
                logger.warning(f"Invalid source span [{start}:{end}]")
                return ""
        else:
            full_text = target_text
            language = target_language
            if not SpanValidator.validate(start, end, target_text):
                logger.warning(f"Invalid target span [{start}:{end}]")
                return ""

        span = full_text[start:end]
        return SpanContextExtractor.extract(
            span=span,
            start=start,
            end=end,
            full_text=full_text,
            language=language,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_cjk_language(language: Optional[str]) -> bool:
        """Check if language is CJK (Chinese, Japanese, Korean)."""
        if not language:
            return False
        return any(cjk in language for cjk in SpanContextExtractor.CJK_LANGUAGES)

    @staticmethod
    def _extract_standard(start: int, end: int, full_text: str) -> str:
        """
        Fixed 4-word context window for space-separated languages.

        Splits the text into tokens, locates the span tokens, then
        slices ±4 words around them.
        """
        n = SpanContextExtractor.CONTEXT_SIZE

        # Build a token list with their character offsets
        tokens = []
        for token in full_text.split(' '):
            tokens.append(token)

        # Reconstruct offsets for each token
        offsets = []
        pos = 0
        for token in tokens:
            offsets.append(pos)
            pos += len(token) + 1  # +1 for the space

        # Find first and last token index that overlaps [start, end)
        first_tok = next(
            (i for i, o in enumerate(offsets) if o >= start),
            len(tokens) - 1
        )
        # Walk back in case start falls inside a token
        while first_tok > 0 and offsets[first_tok] > start:
            first_tok -= 1

        last_tok = first_tok
        for i, o in enumerate(offsets):
            if o < end:
                last_tok = i

        # Apply context window
        ctx_start = max(0, first_tok - n)
        ctx_end   = min(len(tokens), last_tok + n + 1)

        return ' '.join(tokens[ctx_start:ctx_end])

    @staticmethod
    def _extract_cjk(start: int, end: int, full_text: str) -> str:
        """Fixed 4-character context window for CJK languages."""
        n = SpanContextExtractor.CONTEXT_SIZE
        ctx_start = max(0, start - n)
        ctx_end   = min(len(full_text), end + n)
        return full_text[ctx_start:ctx_end]


class SpanValidator:
    """Validate error span positions and content."""

    @staticmethod
    def validate(
        start: int,
        end: int,
        full_text: str,
        min_length: int = 1,
        max_length: Optional[int] = None
    ) -> bool:
        """
        Validate span positions.

        Args:
            start:      Start position.
            end:        End position.
            full_text:  Full text.
            min_length: Minimum span length.
            max_length: Maximum span length (optional).

        Returns:
            True if span is valid, False otherwise.
        """
        if start < 0 or end > len(full_text):
            logger.warning(
                f"Span [{start}:{end}] out of bounds for text length {len(full_text)}"
            )
            return False

        if start >= end:
            logger.warning(f"Invalid span: start ({start}) >= end ({end})")
            return False

        span_length = end - start

        if span_length < min_length:
            logger.warning(f"Span too short: {span_length} < {min_length}")
            return False

        if max_length and span_length > max_length:
            logger.warning(f"Span too long: {span_length} > {max_length}")
            return False

        return True

    @staticmethod
    def extract_span(start: int, end: int, full_text: str) -> Optional[str]:
        """
        Safely extract span from text.

        Args:
            start:     Start position.
            end:       End position.
            full_text: Full text.

        Returns:
            Extracted span or None if invalid.
        """
        if not SpanValidator.validate(start, end, full_text):
            return None
        return full_text[start:end]