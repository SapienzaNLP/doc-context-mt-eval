"""Computes the paper's span-level MPP metric and logs it to WandB."""

import sys
import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import wandb
from mt_metrics_eval import data
from span_mt_metrics_eval import compute as smme_compute

# Add src to path for cross-module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compute_span_level_metrics import compute_span_level_metrics
from config_meta_evaluation import MetaEvaluationConfig
from utils_meta_evaluation import SpanMatchingStats

logger = logging.getLogger(__name__)

SELECTED_DOMAINS = ["news", "literary", "social", "speech"]


def normalize_gold_spans(gold_spans_raw) -> List[Dict[str, Any]]:
    if gold_spans_raw is None:
        return []

    if isinstance(gold_spans_raw, float):
        if pd.isna(gold_spans_raw):
            return []
        return []

    if isinstance(gold_spans_raw, str):
        text = gold_spans_raw.strip()
        if text.lower() in ("", "nan", "none", "null", "-0.0"):
            return []

        try:
            gold_spans_raw = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []

    if not isinstance(gold_spans_raw, list):
        return []

    normalized = []
    for err in gold_spans_raw:
        if not isinstance(err, dict):
            continue

        new_err = dict(err)

        if "start_i" in new_err:
            new_err["start"] = new_err.pop("start_i")
        if "end_i" in new_err:
            new_err["end"] = new_err.pop("end_i")

        normalized.append(new_err)

    return normalized


@dataclass
class SpanCounters:
    """Aggregated counters across all segments."""
    count_inconsistencies: int = 0
    num_segments: int = 0

    smme_pred_spans_all: List[List[Dict]] = field(default_factory=list)
    smme_gold_spans_all: List[List[Dict]] = field(default_factory=list)


class SpanMetricsProcessor:
    """Computes the paper's MPP metric (span-mt-metrics-eval, optimal matching) and logs it to WandB."""

    def __init__(self, config: MetaEvaluationConfig, excluded_domains: Optional[List[str]] = None):
        self.config = config
        self.excluded_domains: Set[str] = set(excluded_domains or [])
        self.wmt_cache: Dict[str, Any] = {}
        self.span_stats = SpanMatchingStats()

    def _get_wmt_data(self, lp: str) -> data.EvalSet:
        if lp not in self.wmt_cache:
            self.wmt_cache[lp] = data.EvalSet(
                self.config.dataset, lp,
                read_stored_metric_scores=True,
                read_stored_ratings=True,
            )
        return self.wmt_cache[lp]

    def _get_domain(self, doc_name: str, domain_field: str = "") -> str:
        """Extract domain from pre-existing domain field or by substring match in doc_name."""
        if domain_field:
            return domain_field.lower().strip()
        for d in SELECTED_DOMAINS:
            if d in doc_name:
                return d
        return "other"

    def _should_skip_entry(self, entry: pd.Series, std_ref: str) -> bool:
        """Check all skip conditions for a single entry."""
        if entry["system"] == std_ref:
            return True

        errors = entry.get("gold_errors", entry.get("para_errors"))
        if errors is None or pd.isna(errors):
            return True
        domain = self._get_domain(
            doc_name=str(entry.get("doc_name", "") or ""),
            domain_field=str(entry.get("domain", "") or ""),
        )
        if domain in self.excluded_domains:
            return True
        return False

    def _build_wandb_table(self) -> wandb.Table:
        return wandb.Table(columns=[
            "System", "Doc Name", "Doc Source", "Doc Translation",
            "Para Source", "Para Reference", "Para Translation",
            "Hypothesis", "Target", "Thinking Content", "Inconsistency",
        ])

    def _save_summary(
        self, scores_dir: str, num_segments: int, count_inconsistencies: int, smme_result
    ) -> None:
        summary_path = f"{scores_dir}/summary_metrics.csv"
        with open(summary_path, "w") as f:
            f.write("===== MPP SUMMARY (span-mt-metrics-eval, optimal matching) =====\n")
            f.write("Metric,Value\n")
            f.write(f"num_segments,{num_segments}\n")
            f.write(f"count_inconsistencies,{count_inconsistencies}\n")
            f.write(f"mpp_precision,{round(smme_result.precision, 4)}\n")
            f.write(f"mpp_recall,{round(smme_result.recall, 4)}\n")
            f.write(f"mpp_f1,{round(smme_result.f_score, 4)}\n")
        logger.info(f"Summary metrics saved to {summary_path}")

    def process(self, all_responses: pd.DataFrame) -> None:
        """Main entry point: parse responses, compute the paper's MPP metric, log to WandB."""
        cfg = self.config
        lp = cfg.language_pair
        language_pairs = all_responses["lang_pair"].unique() if lp == "all" else [lp]

        for lp_single in language_pairs:
            self._get_wmt_data(lp_single)

        counters = SpanCounters()
        table = self._build_wandb_table()
        skipped_domain = 0

        for _, entry in all_responses.iterrows():
            lp_curr = entry["lang_pair"] if lp == "all" else lp
            wmt_data_entry = self._get_wmt_data(lp_curr)

            if self._should_skip_entry(entry, wmt_data_entry.std_ref):
                domain = self._get_domain(
                    doc_name=str(entry.get("doc_name", "") or ""),
                    domain_field=str(entry.get("domain", "") or ""),
                )
                if domain in self.excluded_domains:
                    skipped_domain += 1
                continue

            counters.num_segments += 1
            gold_spans = normalize_gold_spans(entry.get("gold_errors", entry.get("para_errors")))

            metrics = compute_span_level_metrics(
                cfg.model_name, entry["response"], entry["translation"], gold_spans,
                span_stats=self.span_stats,
            )

            counters.count_inconsistencies += metrics["check_inconsistency"]
            counters.smme_pred_spans_all.append(metrics["smme_pred_spans"])
            counters.smme_gold_spans_all.append(metrics["smme_gold_spans"])

            target_spans_str = "\n".join(
                f"{entry['translation'][e['start']:e['end']] if isinstance(entry['translation'], str) and e['start'] != 'missing' else ''} - {e['severity']}"
                for e in gold_spans if e.get("score", 1) != 0 and e.get("is_source_error") is not True
            )

            table.add_data(
                entry["system"], entry["doc_name"],
                entry["doc_source"] if cfg.include_document_context else None,
                entry["doc_translation"] if cfg.include_document_context else None,
                entry["source"],
                entry["reference"] if not cfg.is_reference_less else None,
                entry["translation"],
                entry["response"], target_spans_str, entry["thinking_content"],
                metrics["check_inconsistency"],
            )

        logger.info(
            f"Processed {counters.num_segments} segments, "
            f"skipped {skipped_domain} due to excluded domains {self.excluded_domains}"
        )

        wandb.log({"Error Detection Table": table})

        smme_result = smme_compute(
            counters.smme_pred_spans_all,
            counters.smme_gold_spans_all,
            measure="MPP",
            severity_weights={"minor": 0.2, "major": 1, "critical": 1.0},
            matching_algorithm="optimal",
        )

        wandb.log({
            "Inconsistencies/unparsable_responses": counters.count_inconsistencies,
            "Inconsistencies/unique_match": self.span_stats.unique_match,
            "Inconsistencies/ambiguous_resolved": self.span_stats.ambiguous_resolved,
            "Inconsistencies/ambiguous_fallback": self.span_stats.ambiguous_fallback,
            "Inconsistencies/ambiguous_no_context": self.span_stats.ambiguous_no_context,
            "Inconsistencies/ambiguous_context_not_found": self.span_stats.ambiguous_context_not_found,
            "Inconsistencies/ambiguous_context_not_unique": self.span_stats.ambiguous_context_not_unique,

            "MPP/precision": round(smme_result.precision, 4),
            "MPP/recall": round(smme_result.recall, 4),
            "MPP/f1": round(smme_result.f_score, 4),
        })
        logger.info(
            f"MPP: precision={smme_result.precision:.4f} "
            f"recall={smme_result.recall:.4f} f1={smme_result.f_score:.4f}"
        )

        self._save_summary(
            self.config.scores_dir, counters.num_segments, counters.count_inconsistencies, smme_result
        )
