"""Batch processing and result saving — model-agnostic via model_adapter."""

import logging
from pathlib import Path
from math import ceil
from typing import List, Dict, Any
import pandas as pd
import wandb
import json

from config_evaluation import EvaluationConfig, get_language_names
from prompts_builder_evaluation import PromptGenerator
from responses_generator import ResponseGenerator
from cache import PromptCache
from model_adapter import ModelProfile

logger = logging.getLogger(__name__)


class EvaluationProcessor:
    """Process evaluation batches and save results."""

    def __init__(
        self,
        config: EvaluationConfig,
        generator: ResponseGenerator,
        tokenizer,
        cache: PromptCache,
        profile: ModelProfile,
    ):
        self.config = config
        self.generator = generator
        self.tokenizer = tokenizer
        self.cache = cache
        self.profile = profile
        self.prompt_gen = PromptGenerator()

    # Public API

    def process_all(
        self,
        items: List[Dict[str, Any]],
        model,
        model_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Process all items in batches and return results.

        Args:
            items:      List of dataset items to process.
            model:      The vLLM LLM instance.
            model_name: Name of the model (used for metadata).

        Returns:
            List of response dictionaries.
        """
        all_responses: List[Dict[str, Any]] = []

        self._save_config(model_name)

        source_lang, target_lang = get_language_names(self.config.language_pair)

        self._log_example_prompt(items[0], source_lang, target_lang)

        num_batches = ceil(len(items) / self.config.batch_size)
        logger.info(
            "Processing %d items in %d batches (family=%s)",
            len(items), num_batches, self.profile.family,
        )

        for batch_idx in range(num_batches):
            batch_items = items[
                batch_idx * self.config.batch_size:
                (batch_idx + 1) * self.config.batch_size
            ]

            batch_responses = self._process_batch(
                batch_items, model, model_name, source_lang, target_lang
            )
            all_responses.extend(batch_responses)

            if (batch_idx + 1) % 10 == 0:
                self._save_checkpoint(all_responses, batch_idx + 1, num_batches)

        self._save_final_results(all_responses)
        return all_responses

    # Private helpers

    def _process_batch(
        self,
        batch_items: List[Dict],
        model,
        model_name: str,
        source_lang: str,
        target_lang: str,
    ) -> List[Dict[str, Any]]:
        """Process a single batch of items."""
        prompts = self._generate_prompts(batch_items, source_lang, target_lang)

        # Cache split
        cached_responses: List[tuple] = []
        uncached_prompts: List[str] = []
        uncached_indices: List[int] = []

        for idx, prompt in enumerate(prompts):
            cached = self.cache.load(prompt)
            if cached:
                cached_responses.append((idx, cached))
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(idx)

        # Generate missing responses
        if uncached_prompts:
            logger.info(
                "Generating %d responses (%d from cache)",
                len(uncached_prompts), len(cached_responses),
            )
            new_responses = self.generator.generate(
                uncached_prompts,
                model,
                self.config.enable_thinking,
            )
            for prompt, (thinking, response) in zip(uncached_prompts, new_responses):
                self.cache.save(prompt, response, thinking)
        else:
            logger.info("All %d responses loaded from cache", len(prompts))
            new_responses = []

        # Merge in original order
        all_batch_responses: List[Any] = [None] * len(prompts)
        for idx, cached_response in cached_responses:
            all_batch_responses[idx] = cached_response
        for idx, new_response in zip(uncached_indices, new_responses):
            all_batch_responses[idx] = new_response

        return self._create_response_dicts(
            batch_items, prompts, all_batch_responses, model_name
        )

    def _generate_prompts(
        self,
        items: List[Dict],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """Generate prompts for a batch of items."""
        prompts = []
        for entry in items:
            if self.config.repeat_segment:
                prompt = self.prompt_gen.generate_repeat_prompt(
                    source_lang,
                    target_lang,
                    entry["para_source"],
                    entry["para_translation"],
                    self.config.enable_thinking,
                    self.tokenizer,
                    self.profile,
                )
            elif self.config.include_document_context:
                prompt = self.prompt_gen.generate_document_prompt(
                    source_lang,
                    target_lang,
                    entry["doc_source"],
                    entry["doc_translation"],
                    entry["para_source"],
                    entry["para_translation"],
                    entry["para_reference"],
                    self.config.is_reference_less,
                    self.config.enable_thinking,
                    self.tokenizer,
                    self.profile,     # NEW
                )
            else:
                prompt = self.prompt_gen.generate_paragraph_prompt(
                    source_lang,
                    target_lang,
                    entry["para_source"],
                    entry["para_translation"],
                    entry["para_reference"],
                    self.config.is_reference_less,
                    self.config.enable_thinking,
                    self.tokenizer,
                    self.profile,     # NEW
                )
            prompts.append(prompt)
        return prompts

    def _create_response_dicts(
        self,
        batch_items: List[Dict],
        prompts: List[str],
        responses: List[tuple],
        model_name: str,
    ) -> List[Dict[str, Any]]:
        """Create response dictionaries with metadata."""
        response_dicts = []
        for idx, ((thinking_content, response), entry) in enumerate(
            zip(responses, batch_items)
        ):
            response_dicts.append({
                "lang_pair":               self.config.language_pair,
                "model_family":            self.profile.family,   # NEW
                "domain":                  entry["domain"],
                "prompt":                  prompts[idx],
                "model_name":              model_name,
                "is_reference_less":       self.config.is_reference_less,
                "include_document_context": self.config.include_document_context,
                "repeat_segment":          self.config.repeat_segment,
                "enable_thinking":         self.config.enable_thinking,
                "system":                  entry["system"],
                "para_id":                 entry["para_id"],
                "source":                  entry["para_source"],
                "reference":               entry["para_reference"],
                "translation":             entry["para_translation"],
                "para_errors":             entry["para_errors"],
                "para_score":              entry["para_score"],
                "response":                response,
                "thinking_content":        thinking_content,
                "doc_name":                entry["doc_name"],
                "doc_source":              entry["doc_source"],
                "doc_translation":         entry["doc_translation"],
            })
        return response_dicts

    def _log_example_prompt(
        self,
        example_item: Dict,
        source_lang: str,
        target_lang: str,
    ):
        """Log example prompt to WandB."""
        if self.config.repeat_segment:
            prompt_example = self.prompt_gen.generate_repeat_prompt(
                source_lang,
                target_lang,
                example_item["para_source"],
                example_item["para_translation"],
                self.config.enable_thinking,
                self.tokenizer,
                self.profile,
            )
        elif self.config.include_document_context:
            prompt_example = self.prompt_gen.generate_document_prompt(
                source_lang,
                target_lang,
                example_item["doc_source"],
                example_item["doc_translation"],
                example_item["para_source"],
                example_item["para_translation"],
                example_item["para_reference"],
                self.config.is_reference_less,
                self.config.enable_thinking,
                self.tokenizer,
                self.profile,
            )
        else:
            prompt_example = self.prompt_gen.generate_paragraph_prompt(
                source_lang,
                target_lang,
                example_item["para_source"],
                example_item["para_translation"],
                example_item["para_reference"],
                self.config.is_reference_less,
                self.config.enable_thinking,
                self.tokenizer,
                self.profile,
            )

        wandb.log({
            "Prompt For MQM Error Detection": wandb.Html(
                f'<meta charset="UTF-8"><body><pre>'
                f'{prompt_example.replace("<", "&lt;").replace(">", "&gt;")}'
                f"</pre>"
            )
        })

    def _save_checkpoint(
        self,
        responses: List[Dict],
        batch_num: int,
        total_batches: int,
    ):
        logger.info(
            "Processed %d/%d batches (%d items so far)",
            batch_num, total_batches, len(responses),
        )
        self._save_results(responses)

    def _save_final_results(self, responses: List[Dict]):
        logger.info("Saving final results: %d responses", len(responses))
        self._save_results(responses)

    def _save_results(self, responses: List[Dict]):
        """Save responses to CSV (overwritten on each call)."""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "all_responses.csv"

        try:
            pd.DataFrame(responses).to_csv(output_path, index=False, encoding="utf-8")
            logger.info("Results saved to: %s", output_path)
        except Exception as exc:
            logger.error("Failed to save results: %s", exc)
            raise

    def _save_config(self, model_name: str):
        """Save run configuration as JSON."""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "run_config.txt"

        config_data = self.config.to_log_dict()
        config_data["resolved_model_name"] = model_name
        config_data["model_family"] = self.profile.family

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info("Configuration saved to: %s", config_path)
        except Exception as exc:
            logger.error("Failed to save configuration: %s", exc)
            raise