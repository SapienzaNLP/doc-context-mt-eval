"""Dataset filtering utilities."""

import logging
from typing import Dict, List, Any, Union

try:
    from evaluation.config_evaluation import DataFilterConfig
except ImportError:
    from config_evaluation import DataFilterConfig

logger = logging.getLogger(__name__)

StdRefArg = Union[str, Dict[str, str]]


def _resolve_std_ref(std_ref: StdRefArg, item: dict) -> str:
    """Return the std_ref for this item, supporting both single-LP and multi-LP cases."""
    if isinstance(std_ref, dict):
        return std_ref.get(item.get("lang_pair", ""), "")
    return std_ref


class DatasetFilter:
    """Filter dataset items based on configuration."""

    def __init__(self, config: DataFilterConfig):
        self.config = config

    def filter_items(
        self,
        items: List[Dict[str, Any]],
        std_ref: StdRefArg,
    ) -> List[Dict[str, Any]]:
        """
        Filter dataset items according to configuration.

        std_ref can be a single string (single-LP run) or a dict {lp: std_ref}
        (aggregated run). In the latter case each item is checked against the
        std_ref of its own language pair.
        """
        valid_items = [
            item for item in items
            if self.config.should_include(item, _resolve_std_ref(std_ref, item))
        ]

        logger.info(
            "Filtered dataset: %d total -> %d valid (%.1f%%)",
            len(items), len(valid_items),
            len(valid_items) / len(items) * 100,
        )

        self._log_filter_stats(items, valid_items, std_ref)
        return valid_items

    def _log_filter_stats(
        self,
        all_items: List[Dict],
        valid_items: List[Dict],
        std_ref: StdRefArg,
    ):
        total = len(all_items)
        no_errors = sum(1 for item in all_items if item.get("para_errors") is None)
        is_reference = sum(
            1 for item in all_items
            if item.get("system") == _resolve_std_ref(std_ref, item)
        )
        excluded_domains = sum(
            1 for item in all_items
            if any(d in (item.get("doc_name") or "").lower()
                   for d in self.config.exclude_domains)
        )

        logger.info("Filter statistics:")
        logger.info("Items without errors: %d (%.1f%%)", no_errors, no_errors / total * 100)
        logger.info("Reference translations: %d (%.1f%%)", is_reference, is_reference / total * 100)
        logger.info("Excluded domains: %d (%.1f%%)", excluded_domains, excluded_domains / total * 100)