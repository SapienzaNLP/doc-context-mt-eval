"""Dataset summarization utilities for finetuning data preparation."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FinetuningDatasetSummarizer:
    """
    Generates summary statistics for train/eval datasets produced
    by the finetuning data preparation pipeline.
    """

    @staticmethod
    def generate_split_summary(
        train_records: List[Dict[str, Any]],
        eval_records: List[Dict[str, Any]],
        train_prompts: List[str],
        eval_prompts: List[str],
        tokenizer=None,                        
    ) -> str:
        """
        Generate a comprehensive summary of train and eval splits.

        Args:
            train_records: Raw record dicts for the training split.
            eval_records:  Raw record dicts for the evaluation split.
            train_prompts: Rendered prompt strings for training.
            eval_prompts:  Rendered prompt strings for evaluation.
            tokenizer:     Optional HuggingFace tokenizer for token-level stats.

        Returns:
            Formatted summary string.
        """
        lines: List[str] = []

        def _sep(title: str = "") -> str:
            return ("=" * 80) if not title else f"  {title.upper()}"

        lines += ["", _sep(), "FINETUNING DATASET SUMMARY", _sep(), ""]

        total_records = len(train_records) + len(eval_records)
        lines += [
            _sep("overview"),
            f"  Total records          : {total_records}",
            f"    Train records        : {len(train_records)}"
            f"  ({len(train_records)/total_records*100:.1f}%)",
            f"    Eval  records        : {len(eval_records)}"
            f"  ({len(eval_records)/total_records*100:.1f}%)",
        ]

        # Language pairs — global distribution across both splits
        all_df = pd.DataFrame(train_records + eval_records)
        if "lang_pair" in all_df.columns:
            lines.append(f"\n  Language pairs (global):")
            for lp, count in all_df["lang_pair"].value_counts().items():
                pct = count / total_records * 100
                lines.append(f"    - {lp:<20} {count:>6}  ({pct:.1f}%)")
        lines.append("")

        # Per-split detailed stats
        for split_name, records, prompts in [
            ("TRAIN", train_records, train_prompts),
            ("EVAL",  eval_records,  eval_prompts),
        ]:
            lines.append(_sep(f"{split_name} split"))
            lines += FinetuningDatasetSummarizer._record_stats(records)
            lines += FinetuningDatasetSummarizer._prompt_stats(prompts, tokenizer)
            lines.append("")

        lines += [_sep(), ""]
        return "\n".join(lines)

    @staticmethod
    def _record_stats(records: List[Dict[str, Any]]) -> List[str]:
        """Statistics computed on the raw record dicts."""
        if not records:
            return ["  (no records)"]

        df = pd.DataFrame(records)
        lines: List[str] = []

        # --- Sources / documents -----------------------------------------
        if "para_source" in df.columns:
            lines.append(
                f"  Unique para_source     : {df['para_source'].nunique()}"
            )
        if "doc_source" in df.columns:
            lines.append(
                f"  Unique doc_source      : {df['doc_source'].nunique()}"
            )

        # --- Language pairs (per split) --------------------------------------
        if "lang_pair" in df.columns:
            lines.append(f"  Language pairs:")
            for lp, count in df["lang_pair"].value_counts().items():
                pct = count / len(df) * 100
                lines.append(f"    - {lp:<20} {count:>6}  ({pct:.1f}%)")

        # --- Domains ---------------------------------------------------------
        if "domain" in df.columns:
            lines.append(f"  Domains:")
            for domain, count in df["domain"].value_counts().items():
                pct = count / len(df) * 100
                lines.append(f"    - {domain:<20} {count:>6}  ({pct:.1f}%)")

        # --- Systems ---------------------------------------------------------
        if "system" in df.columns:
            unique_sys = df["system"].nunique()
            lines.append(f"  Systems ({unique_sys} unique):")
            for sys_name in sorted(df["system"].dropna().unique()):
                count = (df["system"] == sys_name).sum()
                pct = count / len(df) * 100
                lines.append(f"    - {sys_name:<20} {count:>6}  ({pct:.1f}%)")

        # --- Error annotations -----------------------------------------------
        if "para_errors" in df.columns:
            non_null   = df["para_errors"].notna().sum()
            null_count = df["para_errors"].isna().sum()
            has_errors = df["para_errors"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            ).sum()
            empty_list = df["para_errors"].apply(
                lambda x: isinstance(x, list) and len(x) == 0
            ).sum()

            lines.append(f"  Error annotations:")
            lines.append(
                f"    Non-null             : {non_null}/{len(df)}"
                f"  ({non_null/len(df)*100:.1f}%)"
            )
            lines.append(f"    Null               : {null_count}")
            lines.append(
                f"    With errors (len>0)  : {has_errors}"
                f"  ({has_errors/len(df)*100:.1f}%)"
            )
            lines.append(f"    Empty list         : {empty_list}")

            if has_errors > 0:
                counts = df["para_errors"].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )
                non_zero = counts[counts > 0]
                non_null_counts = df.loc[
                    df["para_errors"].notna(), "para_errors"
                ].apply(lambda x: len(x) if isinstance(x, list) else 0)

                lines.append(
                    f"    Avg errors (len>0)   : {non_zero.mean():.2f}"
                )
                lines.append(
                    f"    Avg errors (len>=0)  : {non_null_counts.mean():.2f}"
                )
                lines.append(f"    Max errors           : {non_zero.max()}")

        # --- Para scores -----------------------------------------------------
        if "para_score" in df.columns:
            scores = df["para_score"].dropna()
            if not scores.empty:
                lines.append(f"  Para scores:")
                lines.append(
                    f"    Non-null             : {len(scores)}/{len(df)}"
                    f"  ({len(scores)/len(df)*100:.1f}%)"
                )
                lines.append(f"    Mean  : {scores.mean():.4f}")
                lines.append(f"    Std   : {scores.std():.4f}")
                lines.append(
                    f"    Range : [{scores.min():.4f}, {scores.max():.4f}]"
                )

        return lines

    @staticmethod
    def _prompt_stats(prompts: List[str], tokenizer=None) -> List[str]:
        """Statistics computed on the rendered prompt strings."""
        if not prompts:
            return ["  (no prompts)"]

        char_lengths = pd.Series([len(p) for p in prompts])
        lines = [
            f"  Prompt length (chars):"
            f"  mean={char_lengths.mean():.0f}"
            f"  median={char_lengths.median():.0f}"
            f"  min={char_lengths.min()}"
            f"  max={char_lengths.max()}",
        ]

        if tokenizer is not None:
            token_lengths = pd.Series([
                len(ids) for ids in
                tokenizer(prompts, add_special_tokens=False)["input_ids"]
            ])
            lines += [
                f"  Prompt length (tokens):"
                f"  mean={token_lengths.mean():.0f}"
                f"  median={token_lengths.median():.0f}"
                f"  p95={token_lengths.quantile(0.95):.0f}"
                f"  p99={token_lengths.quantile(0.99):.0f}"
                f"  max={token_lengths.max()}",
                f"  Samples > 4096 tokens: {(token_lengths > 4096).sum()}"
                f"  ({(token_lengths > 4096).mean()*100:.1f}%)",
                f"  Samples > 8192 tokens: {(token_lengths > 8192).sum()}"
                f"  ({(token_lengths > 8192).mean()*100:.1f}%)",
            ]

        return lines

    @staticmethod
    def save_summary(
        summary: str,
        save_path: Path,
        filename: str = "dataset_summary.txt",
    ) -> None:
        """Write the summary string to a .txt file inside save_path."""
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / filename
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info(f"Dataset summary saved to {output_file}")
        except IOError as e:
            logger.error(f"Failed to save summary to {output_file}: {e}")
            raise