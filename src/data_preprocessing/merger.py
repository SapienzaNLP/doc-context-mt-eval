"""Dataset merger for WMT evaluation data."""

import logging
from typing import List, Dict, Any, Optional, Tuple

from mt_metrics_eval import data

from config_preprocessing import ProcessingConfig
from builders import ContextWindowBuilder
from utils_preprocessing import RatingSerializer

logger = logging.getLogger(__name__)


class DatasetMerger:
    """Merges WMT evaluation data into unified format."""

    def __init__(
        self,
        wmt: data.EvalSet,
        index_to_doc: Dict[int, Dict[str, Any]],
        mqm_mapping: Optional[Dict[Tuple, Tuple]],
        config: ProcessingConfig
    ):
        self.wmt = wmt
        self.index_to_doc = index_to_doc
        self.mqm_mapping = mqm_mapping
        self.config = config

        self.ratings, self.scores = self._load_ratings_and_scores()

    def _load_ratings_and_scores(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Load MQM ratings and scores."""
        try:
            ratings = self.wmt.Ratings(rating_name="mqm.merged")
            scores = self.wmt.Scores(level="seg", scorer="mqm")
            logger.info("Loaded MQM ratings successfully")
            return ratings, scores
        except Exception as e:
            logger.warning(f"Could not load MQM ratings: {e}")
            return None, None

    def _normalize_spans(self, spans) -> Optional[list]:
        """Normalize start_i/end_i keys to start/end."""
        if not spans:
            return spans
        normalized = []
        for err in spans:
            new_err = dict(err)
            if 'start_i' in new_err:
                new_err['start'] = new_err.pop('start_i')
            if 'end_i' in new_err:
                new_err['end'] = new_err.pop('end_i')
            normalized.append(new_err)
        return normalized

    def merge(self) -> List[Dict[str, Any]]:
        """Create merged dataset from WMT evaluation data."""
        merged_dataset = []

        for sys_name in sorted(self.wmt.sys_outputs.keys()):
            sys_outputs = self.wmt.sys_outputs[sys_name]

            sys_ratings = (
                self.ratings[sys_name] if self.ratings is not None
                else [None] * len(sys_outputs)
            )
            sys_scores = (
                self.scores[sys_name] if self.scores is not None
                else [None] * len(sys_outputs)
            )

            for i in range(len(sys_outputs)):
                entry = self._create_entry(
                    sys_name, i, sys_outputs, sys_ratings, sys_scores
                )
                if entry:
                    merged_dataset.append(entry)

        logger.info(f"Created {len(merged_dataset)} merged entries")
        return merged_dataset

    def _create_entry(
        self,
        sys_name: str,
        i: int,
        sys_outputs: List[str],
        sys_ratings: List,
        sys_scores: List
    ) -> Optional[Dict[str, Any]]:
        """Create a single dataset entry for system and segment index."""
        if i not in self.index_to_doc:
            logger.warning(f"Index {i} not found in document mapping")
            return None

        doc_info = self.index_to_doc[i]
        start, end = doc_info["doc_indices"]

        if self.config.enable_context_windows:
            window_indices = ContextWindowBuilder.get_window_indices(
                i, start, end, self.config.n_pre, self.config.n_post
            )
        else:
            window_indices = list(range(start, end))

        doc_source = "\n\n".join(
            self.wmt.src[j] for j in window_indices
            if j < len(self.wmt.src)
        )
        doc_translation = "\n\n".join(
            sys_outputs[j] for j in window_indices
            if j < len(sys_outputs)
        )

        source = self.wmt.src[i] if i < len(self.wmt.src) else ""
        translation = sys_outputs[i] if i < len(sys_outputs) else ""
        reference = (
            self.wmt.all_refs[self.wmt.std_ref][i]
            if i < len(self.wmt.all_refs[self.wmt.std_ref])
            else ""
        )

        entry = {
            "lang_pair": self.wmt.lp,
            "domain": doc_info['domain'],
            "system": sys_name,
            "para_id": i,
            "doc_name": doc_info["doc_name"],
            "doc_indices": window_indices,
            "doc_source": doc_source,
            "doc_translation": doc_translation,
            "para_score": sys_scores[i] if i < len(sys_scores) else None,
        }

        self._add_mqm_data(
            entry, sys_name, i, doc_info,
            source, reference, translation, sys_ratings
        )

        return entry

    def _add_mqm_data(
        self,
        entry: Dict,
        sys_name: str,
        i: int,
        doc_info: Dict,
        source: str,
        reference: str,
        translation: str,
        sys_ratings: List
    ):
        """Add MQM-specific data to entry (errors and text segments)."""
        if self.mqm_mapping:
            key = (sys_name, i, doc_info['doc_name'])
            mqm_data = self.mqm_mapping.get(key)

            if mqm_data:
                gold_spans, src, ref, hyp, score = mqm_data
                if score is not None:
                    entry['para_score'] = score
                entry['para_source'] = src
                entry['para_reference'] = ref
                entry['para_translation'] = hyp
                entry["para_errors"] = self._normalize_spans(gold_spans)
            else:
                entry['para_source'] = source
                entry['para_reference'] = reference
                entry['para_translation'] = translation
                entry["para_errors"] = None
        else:
            entry['para_source'] = source
            entry['para_reference'] = reference
            entry['para_translation'] = translation
            entry["para_errors"] = RatingSerializer.serialize(
                sys_ratings[i] if i < len(sys_ratings) else None
            )