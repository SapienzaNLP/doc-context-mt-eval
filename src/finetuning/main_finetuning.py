"""Main script for finetuning MT evaluation models."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed.fsdp
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import concatenate_datasets

# Compat shim: transformers' Trainer unconditionally calls this FSDP2 API (added to
# fix a `generate()` + LoRA + FSDP2 DTensor bug) whenever FSDP is enabled, regardless
# of whether generation is actually used. Older torch versions predate the function;
# we use classic FSDP1 (not FSDP2) and never call `generate()` during eval (no
# predict_with_generate), so a no-op stub is safe.
if not hasattr(torch.distributed.fsdp, "register_fsdp_forward_method"):
    torch.distributed.fsdp.register_fsdp_forward_method = lambda module, method_name: None

from config_finetuning import (
    FinetuningConfig,
    ModelConfig,
    DataConfig,
    LoRAConfig,
    SeedConfig,
    PromptTemplateConfig
)
from data_preparation_finetuning import DataPreparator
from prompt_builder_finetuning import PromptBuilder
from lora_config import apply_lora
from trainer import ModelTrainer
from utils_finetuning import get_language_names, setup_cuda

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> FinetuningConfig:
    """Parse command-line arguments and create configuration."""
    parser = argparse.ArgumentParser(
        description="Finetune models for MT quality evaluation"
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or path"
    )

    # Data parameters
    parser.add_argument(
        "--data_paths",
        type=str,
        required=True,
        help="Comma-separated list of paths to preprocessed JSONL data"
    )
    parser.add_argument(
        "--lp",
        type=str,
        required=True,
        help="Comma-separated list of language pairs (e.g., en-de,zh-en)"
    )
    parser.add_argument(
        "--exclude_domains",
        type=str,
        help="Comma-separated domains to exclude",
        default="speech"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data for evaluation"
    )

    # Template parameters
    template_group = parser.add_mutually_exclusive_group(required=True)
    template_group.add_argument(
        "--use_all_templates",
        action="store_true",
        help="Generate all 4 template variations (para/doc × ref-based/ref-less)"
    )
    template_group.add_argument(
        "--use_single_template",
        action="store_true",
        help="Generate only one template variation"
    )
    parser.add_argument(
        "--include_document_context",
        type=str,
        choices=["yes", "no"],
        help="Include document context (required if --use_single_template)"
    )
    parser.add_argument(
        "--is_reference_less",
        type=str,
        choices=["yes", "no"],
        help="Reference-less evaluation (required if --use_single_template)"
    )

    # Training parameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")

    # LoRA parameters (defaults match paper Table 9)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Output parameters
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")

    parser.add_argument(
        "--finetuning_data_dir",
        type=str,
        default="src/finetuning/finetuning_data",
        help="Directory to save finetuning datasets"
    )

    # WandB parameters
    parser.add_argument("--wandb_project_name", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="WandB run name")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline"],
        default="online",
        help="WandB mode"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Validate template arguments
    if args.use_single_template:
        if args.include_document_context is None or args.is_reference_less is None:
            parser.error(
                "--use_single_template requires both "
                "--include_document_context and --is_reference_less"
            )

    # Parse comma-separated values
    data_paths = [Path(p.strip()) for p in args.data_paths.split(",")]
    language_pairs = [lp.strip() for lp in args.lp.split(",")]
    exclude_domains = (
        [d.strip() for d in args.exclude_domains.split(",")]
        if args.exclude_domains else []
    )

    # Validate matching counts
    if len(data_paths) != len(language_pairs):
        parser.error(
            f"Number of data paths ({len(data_paths)}) must match "
            f"number of language pairs ({len(language_pairs)}). "
            "Provide one data file per language pair in the same order."
        )

    # Validate language pair format
    for lp in language_pairs:
        if '-' not in lp or len(lp.split('-')) != 2:
            parser.error(
                f"Invalid language pair format: '{lp}'. "
                "Expected format: 'source-target' (e.g., 'en-de')"
            )

    # Build config objects
    model_config_kwargs = {"model_name": args.model_name}
    if args.learning_rate is not None:
        model_config_kwargs["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        model_config_kwargs["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        model_config_kwargs["train_batch_size"] = args.batch_size
        model_config_kwargs["eval_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        model_config_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_length is not None:
        model_config_kwargs["max_length"] = args.max_length
    model_config = ModelConfig(**model_config_kwargs)

    data_config = DataConfig(
        data_paths=data_paths,
        exclude_domains=exclude_domains,
        language_pairs=language_pairs,
        test_size=args.test_size
    )

    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    seed_config = SeedConfig(seed=args.seed)

    if args.use_all_templates:
        template_config = PromptTemplateConfig(use_all_templates=True)
    else:
        template_config = PromptTemplateConfig(
            include_document_context=(args.include_document_context == "yes"),
            is_reference_less=(args.is_reference_less == "yes"),
            use_all_templates=False
        )

    return FinetuningConfig(
        model_config=model_config,
        data_config=data_config,
        lora_config=lora_config,
        seed_config=seed_config,
        template_config=template_config,
        language_pairs=language_pairs,
        output_dir=args.output_dir,
        finetuning_data_dir=args.finetuning_data_dir,
        wandb_project_name=args.wandb_project_name,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode
    )


def _make_wandb_config(config: FinetuningConfig) -> dict:
    """Convert FinetuningConfig to WandB-serializable dictionary."""
    return {
        # Model
        "model_name": config.model_config.model_name,
        "learning_rate": config.model_config.learning_rate,
        "num_epochs": config.model_config.num_epochs,
        "train_batch_size": config.model_config.train_batch_size,
        "eval_batch_size": config.model_config.eval_batch_size,
        "gradient_accumulation_steps": config.model_config.gradient_accumulation_steps,
        "max_length": config.model_config.max_length,
        "torch_dtype": str(config.model_config.torch_dtype),
        "device_map": config.model_config.device_map,
        # Data
        "data_paths": [str(p) for p in config.data_config.data_paths],
        "exclude_domains": config.data_config.exclude_domains,
        "language_pairs": config.data_config.language_pairs,
        "test_size": config.data_config.test_size,
        # LoRA
        "lora_r": config.lora_config.r,
        "lora_alpha": config.lora_config.lora_alpha,
        "lora_dropout": config.lora_config.lora_dropout,
        "lora_target_modules": config.lora_config.target_modules,
        # Template
        "use_all_templates": config.template_config.use_all_templates,
        "include_document_context": config.template_config.include_document_context,
        "is_reference_less": config.template_config.is_reference_less,
        # Other
        "seed": config.seed_config.seed,
        "output_dir": config.output_dir,
        "wandb_project_name": config.wandb_project_name,
        "wandb_run_name": config.wandb_run_name,
    }


def save_prompt_examples(
    train_dataset,
    eval_dataset,
    output_dir: Path,
    tokenizer,
    num_examples: int = 5
):
    """Save example prompts to text files and log to WandB."""
    examples_dir = output_dir / "prompt_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {num_examples} prompt examples to {examples_dir}")

    def _collect_and_save(dataset, prefix: str) -> list:
        examples = []
        for i in range(min(num_examples, len(dataset))):
            example = dataset[i]["text"]
            examples.append(example)
            example_file = examples_dir / f"{prefix}_example_{i+1}.txt"
            with open(example_file, 'w', encoding='utf-8') as f:
                f.write(f"=== {prefix.upper()} EXAMPLE {i+1} ===\n\n")
                f.write(example)
                f.write("\n\n" + "=" * 80 + "\n")
            logger.info(f"Saved {prefix} example {i+1} to {example_file}")
        return examples

    train_examples = _collect_and_save(train_dataset, "train")
    eval_examples  = _collect_and_save(eval_dataset,  "eval")

    # Save combined file
    combined_file = examples_dir / "all_examples.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for section_label, examples in [
            ("TRAINING EXAMPLES", train_examples),
            ("EVALUATION EXAMPLES", eval_examples),
        ]:
            f.write("=" * 80 + "\n")
            f.write(f"{section_label}\n")
            f.write("=" * 80 + "\n\n")
            for i, example in enumerate(examples, 1):
                f.write(f"\n{'=' * 80}\n{section_label[:-1]} {i}\n{'=' * 80}\n\n")
                f.write(example)
                f.write("\n\n")
    logger.info(f"Saved combined examples to {combined_file}")

    # WandB logging
    if wandb.run is not None:
        for i, ex in enumerate(train_examples, 1):
            wandb.log({f"train_example_{i}": wandb.Html(f"<pre>{ex}</pre>")}, step=0)
        for i, ex in enumerate(eval_examples, 1):
            wandb.log({f"eval_example_{i}": wandb.Html(f"<pre>{ex}</pre>")}, step=0)

        artifact = wandb.Artifact(
            name="prompt_examples",
            type="dataset_samples",
            description="Example prompts from training and evaluation datasets"
        )
        artifact.add_dir(str(examples_dir))
        wandb.log_artifact(artifact)

        wandb.log({
            "training_examples": wandb.Table(
                columns=["example_id", "prompt_length", "prompt_text"],
                data=[[i+1, len(ex), ex[:500] + "..." if len(ex) > 500 else ex]
                      for i, ex in enumerate(train_examples)]
            ),
            "evaluation_examples": wandb.Table(
                columns=["example_id", "prompt_length", "prompt_text"],
                data=[[i+1, len(ex), ex[:500] + "..." if len(ex) > 500 else ex]
                      for i, ex in enumerate(eval_examples)]
            ),
        }, step=0)
        logger.info("Logged examples to WandB")

    # Token length statistics (sampled)
    train_lengths = [
        len(tokenizer(train_dataset[i]["text"], add_special_tokens=False)["input_ids"])
        for i in range(len(train_dataset))
    ]
    eval_lengths = [
        len(tokenizer(eval_dataset[i]["text"], add_special_tokens=False)["input_ids"])
        for i in range(len(eval_dataset))
    ]
    stats = {
        "train_avg_token_length": sum(train_lengths) / len(train_lengths),
        "train_max_token_length": max(train_lengths),
        "train_min_token_length": min(train_lengths),
        "eval_avg_token_length":  sum(eval_lengths)  / len(eval_lengths),
        "eval_max_token_length":  max(eval_lengths),
        "eval_min_token_length":  min(eval_lengths),
    }
    if wandb.run is not None:
        wandb.log(stats, step=0)
    logger.info(f"Prompt token statistics: {stats}")


def main():
    """Main entry point."""
    config = parse_arguments()

    setup_cuda()
    config.seed_config.apply()
    logger.info(f"Random seed set to: {config.seed_config.seed}")

    wandb_dir = os.environ.get("WANDB_DIR", "logs/wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_MODE"] = config.wandb_mode
    if config.wandb_mode != "offline":
        wandb.login()
    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_name,
        config=_make_wandb_config(config),
        dir=wandb_dir
    )

    # Save full run configuration to file
    config.save_run_config(Path(config.output_dir))

    # Load model and tokenizer
    logger.info(f"Loading model: {config.model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_config.model_name,
        torch_dtype=config.model_config.torch_dtype,
        # "eager" (not "sdpa"): unlike SDPA's "Math" fallback, eager's softmax is
        # upcast to fp32 regardless of model dtype, avoiding bf16 accumulation
        # blowups over long (8192-token) sequences. flash_attention_2 would also
        # fix it but isn't usable here (glibc incompatibility).
        attn_implementation="eager",
        device_map=None
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # per-language-pair data preparation
    all_train_datasets = []
    all_eval_datasets  = []

    logger.info(f"Processing {len(config.language_pairs)} language pair(s)...")

    for lp, data_path in zip(config.language_pairs, config.data_config.data_paths):
        source_lang, target_lang = get_language_names(lp)
        logger.info(f"Processing language pair: {source_lang} → {target_lang} ({lp})")
        logger.info(f"  Using data file: {data_path}")

        lp_data_config = DataConfig(
            data_paths=[data_path],
            exclude_domains=config.data_config.exclude_domains,
            language_pairs=[lp],
            test_size=config.data_config.test_size
        )

        data_preparator = DataPreparator(
            lp_data_config,
            config.seed_config,
            config.template_config,
            PromptBuilder(tokenizer),
            source_lang,
            target_lang,
        )

        # Each lp is saved in its own subdirectory to avoid overwriting
        train_dataset, eval_dataset = data_preparator.prepare(
            Path(config.finetuning_data_dir) / Path(data_path).stem,
            tokenizer = tokenizer
        )

        all_train_datasets.append(train_dataset)
        all_eval_datasets.append(eval_dataset)
        logger.info(f"  {lp}: {len(train_dataset)} train, {len(eval_dataset)} eval samples")

    # combine & shuffle
    combined_train = concatenate_datasets(all_train_datasets)
    combined_eval  = concatenate_datasets(all_eval_datasets)
    logger.info(
        f"\nCombined dataset (before shuffle): "
        f"{len(combined_train)} train, {len(combined_eval)} eval"
    )

    logger.info("Shuffling combined datasets to mix language pairs...")
    combined_train = combined_train.shuffle(seed=config.seed_config.seed)
    combined_eval  = combined_eval.shuffle(seed=config.seed_config.seed)
    logger.info(
        f"Combined dataset (after shuffle): "
        f"{len(combined_train)} train, {len(combined_eval)} eval"
    )

    # save combined datasets (used by the trainer)
    DataPreparator.save_combined_datasets(
        combined_train,
        combined_eval,
        save_path=Path(config.finetuning_data_dir) / "combined",
    )

    # prompt examples
    save_prompt_examples(
        combined_train,
        combined_eval,
        Path(config.output_dir),
        tokenizer=tokenizer,
        num_examples=5
    )

    # LoRA + training
    logger.info("Applying LoRA configuration...")
    model = apply_lora(model, config.lora_config)

    logger.info("Initializing trainer...")
    trainer = ModelTrainer(config)

    trainer.train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_train,
        eval_dataset=combined_eval
    )

    logger.info("\nFinetuning completed successfully!\n")


if __name__ == "__main__":
    main()