"""Main script for MT evaluation using LLMs — supports Qwen3 and Gemma 4."""

import argparse
import logging
import os

import wandb
from transformers import AutoTokenizer
from vllm import LLM
from mt_metrics_eval import data

from config_evaluation import (
    EvaluationConfig,
    ModelConfig,
    SeedConfig,
    DataFilterConfig,
)
from model_adapter import get_model_profile
from filters import DatasetFilter
from responses_generator import ResponseGenerator
from spans_generator import EvaluationProcessor
from cache import PromptCache
from utils_evaluation import load_dataset, setup_cuda

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> EvaluationConfig:
    """Parse command-line arguments and create configuration."""
    parser = argparse.ArgumentParser(description="Evaluate MT quality using LLMs")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name or local path. "
             "Supported families: Qwen3, Gemma 4.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=32768)

    # Dataset
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lp", type=str, required=True)

    # Evaluation mode
    parser.add_argument(
        "--is_reference_less", type=str, required=True, choices=["yes", "no"]
    )
    parser.add_argument(
        "--include_document_context", type=str, required=True, choices=["yes", "no"]
    )
    parser.add_argument(
        "--repeat_segment", type=str, choices=["yes", "no"], default="no",
        help="Reference-less only: duplicate the segment blocks instead of "
             "showing the full document (repetition control).",
    )
    parser.add_argument(
        "--enable_thinking", type=str, required=True, choices=["yes", "no"]
    )

    # Processing
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # Cache
    parser.add_argument(
        "--enable_cache", type=str, choices=["yes", "no"], default="no"
    )

    # Filters
    parser.add_argument(
        "--exclude_domains", type=str, default="speech",
        help="Comma-separated list of domains to exclude.",
    )

    # WandB
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--wandb_project_name", type=str, required=True)
    parser.add_argument(
        "--wandb_mode", type=str, choices=["online", "offline"], default="online"
    )

    args = parser.parse_args()

    if args.repeat_segment == "yes":
        if args.include_document_context == "yes":
            parser.error("--repeat_segment yes is incompatible with --include_document_context yes.")
        if args.is_reference_less == "no":
            parser.error("--repeat_segment yes requires --is_reference_less yes.")

    model_config = ModelConfig(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_new_tokens=args.max_new_tokens,
    )
    seed_config = SeedConfig(seed=args.seed)
    filter_config = DataFilterConfig(
        exclude_domains=[d.strip() for d in args.exclude_domains.split(",")]
    )

    evaluation_config = EvaluationConfig(
        dataset=args.dataset,
        language_pair=args.lp,
        data_path=args.data_path,
        is_reference_less=(args.is_reference_less == "yes"),
        include_document_context=(args.include_document_context == "yes"),
        repeat_segment=(args.repeat_segment == "yes"),
        enable_thinking=(args.enable_thinking == "yes"),
        batch_size=args.batch_size,
        wandb_project_name=args.wandb_project_name,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        model_config=model_config,
        seed_config=seed_config,
        filter_config=filter_config,
    )

    evaluation_config.cache_dir = (
        f"prompt_cache/{args.dataset}/{args.lp}/{args.model_name}"
    )
    evaluation_config.enable_cache = args.enable_cache == "yes"

    return evaluation_config


def main():
    """Main entry point."""
    config = parse_arguments()

    # CUDA & seed
    setup_cuda()
    config.seed_config.apply()
    logger.info("Random seed set to: %d", config.seed_config.seed)

    # Model profile (family-specific settings)
    profile = get_model_profile(config.model_config.model_name)
    logger.info(
        "Model profile: family=%s | max_model_len=%d | thinking_end='%s'",
        profile.family, profile.max_model_len, profile.thinking_end_token,
    )

    # Override ModelConfig sampling params with profile defaults when the user
    # hasn't explicitly set them (i.e. when they still hold Qwen3 defaults).
    # This ensures correct out-of-the-box behaviour for Gemma 4 without
    # requiring the user to re-specify every sampling parameter.
    config.model_config.thinking_temperature = profile.thinking_temperature
    config.model_config.thinking_top_p       = profile.thinking_top_p
    config.model_config.thinking_top_k       = profile.thinking_top_k
    config.model_config.thinking_min_p       = profile.thinking_min_p

    # Load vLLM model
    logger.info("Loading model: %s", config.model_config.model_name)
    model = LLM(
        model=config.model_config.model_name,
        dtype=config.model_config.dtype,
        seed=config.seed_config.seed,
        tensor_parallel_size=config.model_config.tensor_parallel_size,
        # Use the profile's context length instead of a hard-coded value.
        max_model_len=profile.max_model_len,
        max_num_seqs=32,
        gpu_memory_utilization=0.85,
        # Disable multimodal slots — this pipeline is text-only.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_config.model_name)
    tokenizer.padding_side = "left"

    # Dataset
    logger.info("Loading dataset: %s", config.dataset_path)
    all_items = load_dataset(config.dataset_path)

    wmt = data.EvalSet(
        config.dataset,
        config.language_pair,
        read_stored_metric_scores=True,
        read_stored_ratings=True,
    )

    dataset_filter = DatasetFilter(config.filter_config)
    valid_items = dataset_filter.filter_items(all_items, wmt.std_ref)

    if not valid_items:
        logger.error("No valid items found after filtering!")
        return

    # WandB
    os.environ["WANDB_MODE"] = config.wandb_mode
    wandb.login()

    config_dict = vars(config).copy()
    if "model_config" in config_dict and hasattr(config_dict["model_config"], "dtype"):
        mc = vars(config_dict["model_config"]).copy()
        mc["dtype"] = str(mc["dtype"])
        config_dict["model_config"] = mc
    config_dict["model_family"] = profile.family   # log the detected family

    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_name,
        config=config_dict,
        reinit=True,
        dir="logs/wandb",
    )

    # Cache
    cache = PromptCache(cache_dir=config.cache_dir, enabled=config.enable_cache)
    cache_stats = cache.get_stats()
    logger.info("Cache stats: %s", cache_stats)
    wandb.log({"cache_stats": cache_stats})

    # Pipeline components
    # Pass the profile to ResponseGenerator so it can parse thinking correctly.
    generator = ResponseGenerator(config.model_config, config.seed_config, profile)

    processor = EvaluationProcessor(
        config,
        generator,
        tokenizer,
        cache,
        profile,   # NEW: processor needs profile to build prompts correctly
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    all_responses = processor.process_all(
        valid_items,
        model,
        config.model_config.model_name,
    )

    final_cache_stats = cache.get_stats()
    logger.info("Final cache stats: %s", final_cache_stats)
    wandb.log({"final_cache_stats": final_cache_stats})

    wandb.finish()
    logger.info("Evaluation completed — total responses: %d", len(all_responses))


if __name__ == "__main__":
    main()