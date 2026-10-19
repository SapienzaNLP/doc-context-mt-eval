"""Main entry point for meta-evaluation pipeline."""

import argparse
import logging
from pathlib import Path

import sys
import os

import pandas as pd
import wandb
from mt_metrics_eval import data as wmt_data

# Add src to path for cross-module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_evaluation.config_meta_evaluation import MetaEvaluationConfig, WandBConfig
from meta_evaluation.span_metrics_processor import SpanMetricsProcessor
from evaluation.config_evaluation import DataFilterConfig
from evaluation.filters import DatasetFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> tuple[MetaEvaluationConfig, DataFilterConfig]:
    parser = argparse.ArgumentParser(description="Meta-evaluation of MT span metrics")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lp", type=str, required=True)
    parser.add_argument("--is_reference_less", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--include_document_context", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--are_aggregated_results", type=str, default="no", choices=["yes", "no"])
    parser.add_argument("--ratings_level", type=str, default="mqm")
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--wandb_project_name", type=str, required=True)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--wandb_new_run", type=str, choices=["yes", "no"], default="yes")

    # Filter arguments
    parser.add_argument(
        "--exclude_domains",
        type=str,
        default="",
        help="Comma-separated list of additional domains to exclude (e.g. 'news,literary').",
    )
    parser.add_argument(
        "--exclude_speech",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Whether to exclude the 'speech' domain (default: yes).",
    )

    args = parser.parse_args()

    # Build exclude_domains list
    extra_domains = [d.strip() for d in args.exclude_domains.split(",") if d.strip()]
    if args.exclude_speech == "yes":
        extra_domains = list(set(extra_domains + ["speech"]))

    filter_config = DataFilterConfig(exclude_domains=extra_domains)

    meta_config = MetaEvaluationConfig(
        dataset=args.dataset,
        language_pair=args.lp,
        model_name=args.model_name,
        is_reference_less=(args.is_reference_less == "yes"),
        include_document_context=(args.include_document_context == "yes"),
        are_aggregated_results=(args.are_aggregated_results == "yes"),
        ratings_level=args.ratings_level,
        wandb=WandBConfig(
            project_name=args.wandb_project_name,
            run_name=args.wandb_run_name,
            mode=args.wandb_mode,
            new_run=(args.wandb_new_run == "yes"),
        ),
    )

    return meta_config, filter_config


def init_wandb(config: MetaEvaluationConfig) -> None:
    os.environ["WANDB_MODE"] = config.wandb.mode

    if config.wandb.new_run:
        wandb.init(
            project=config.wandb.project_name,
            name=config.wandb.run_name,
            dir="logs/wandb",
        )
    else:
        api = wandb.Api()
        runs = api.runs(config.wandb.project_name)
        run_to_resume = next((r for r in runs if r.name == config.wandb.run_name), None)
        if run_to_resume is None:
            raise ValueError(f"WandB run '{config.wandb.run_name}' not found.")
        wandb.init(
            project=config.wandb.project_name,
            id=run_to_resume.id,
            resume="allow",
            dir="logs/wandb",
        )


def load_responses(config: MetaEvaluationConfig) -> pd.DataFrame:
    import csv as _csv
    import sys as _sys

    _csv.field_size_limit(_sys.maxsize)

    path = Path(config.responses_path).expanduser()
    logger.info(f"Loading responses from: {path}")
    try:
        df = pd.read_csv(path, engine="python")
    except pd.errors.ParserError:
        logger.warning(f"CSV truncated, reading partial data from: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            reader = _csv.reader(f)
            rows: list[list[str]] = []
            try:
                for row in reader:
                    rows.append(row)
            except _csv.Error:
                pass
        df = pd.DataFrame(rows[1:], columns=rows[0]) if rows else pd.DataFrame()
    if 'lang_pair' not in df.columns:
        df['lang_pair'] = config.language_pair
    logger.info(f"Loaded {len(df)} rows")
    return df


def get_std_refs(config: MetaEvaluationConfig, all_responses: pd.DataFrame) -> dict:
    """Return {lp: std_ref} for every LP present in all_responses."""
    lps = (
        all_responses["lang_pair"].unique().tolist()
        if config.language_pair == "all"
        else [config.language_pair]
    )
    result = {}
    for lp in lps:
        wmt = wmt_data.EvalSet(
            config.dataset, lp,
            read_stored_metric_scores=False,
            read_stored_ratings=False,
        )
        result[lp] = wmt.std_ref
    return result


def apply_filters(
    all_responses: pd.DataFrame,
    filter_config: DataFilterConfig,
    std_refs: dict,
) -> pd.DataFrame:
    """Convert DataFrame rows to dicts, filter with per-LP std_ref, then rebuild DataFrame."""
    items = all_responses.to_dict(orient="records")
    dataset_filter = DatasetFilter(filter_config)
    valid_items = dataset_filter.filter_items(items, std_refs)

    if not valid_items:
        raise ValueError("No valid items remaining after filtering.")

    return pd.DataFrame(valid_items)


def main():
    config, filter_config = parse_arguments()

    init_wandb(config)
    logger.info(f"WandB initialized: project={config.wandb.project_name}, run={config.wandb.run_name}")

    all_responses = load_responses(config)
    logger.info(f"Loaded {len(all_responses)} responses")

    # Apply filters
    std_refs = get_std_refs(config, all_responses)
    logger.info(f"Applying filters (exclude_domains={filter_config.exclude_domains})")
    all_responses = apply_filters(all_responses, filter_config, std_refs)
    logger.info(f"Remaining segments after filtering: {len(all_responses)}")
    # logger.info(f"Domains present AFTER filter: {all_responses['domain'].unique().tolist()}")

    # Log filter config to WandB
    wandb.config.update({
        "exclude_domains": filter_config.exclude_domains,
        "exclude_speech": "speech" in filter_config.exclude_domains,
    })

    # --- Span-level metrics ---
    span_processor = SpanMetricsProcessor(config, excluded_domains=filter_config.exclude_domains)
    span_processor.process(all_responses)
    logger.info("Span-level metrics computed and logged.")

    wandb.finish()
    logger.info("Meta-evaluation completed successfully.")


if __name__ == "__main__":
    main()