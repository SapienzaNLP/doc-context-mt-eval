"""Data preparation for finetuning."""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from config_finetuning import DataConfig, SeedConfig, PromptTemplateConfig
from prompt_builder_finetuning import PromptBuilder
from data_summarizer_finetuning import FinetuningDatasetSummarizer

logger = logging.getLogger(__name__)

# Mapping from raw/noisy error type strings to canonical prompt categories.
# Canonical categories (as stated in the prompt):
#   accuracy:         addition, mistranslation, omission, untranslated text
#   fluency:          character encoding, grammar, inconsistency, punctuation, register, spelling
#   locale convention: currency, date, name, telephone, time
#   style:            awkward
#   terminology:      inappropriate for context, inconsistent use
#   non-translation
#   other
ERROR_TYPE_MAPPING: dict = {
    # accuracy
    "mistranslation":                   "accuracy/mistranslation",
    "wrong_named_entity":               "accuracy/mistranslation",
    "mt_hallucination":                 "accuracy/mistranslation",
    "omission":                         "accuracy/omission",
    "addition":                         "accuracy/addition",
    "untranslated":                     "accuracy/untranslated text",
    "do_not_translate":                 "accuracy/untranslated text",
    "accuracy/source language fragment":"accuracy/untranslated text",
    "accuracy/untranslated":            "accuracy/untranslated text",

    # fluency
    "grammar":                          "fluency/grammar",
    "agreement":                        "fluency/grammar",
    "word_order":                       "fluency/grammar",
    "spelling":                         "fluency/spelling",
    "capitalization":                   "fluency/spelling",
    "punctuation":                      "fluency/punctuation",
    "whitespace":                       "fluency/punctuation",
    "register":                         "fluency/register",
    "inconsistency":                    "fluency/inconsistency",
    "markup_tag":                       "fluency/punctuation",
    "fluency/display":                  "other",

    # style
    "style/unnatural or awkward":       "style/awkward",
    "style/bad sentence structure":     "style/awkward",
    "unnatural_flow":                   "style/awkward",
    "lacks_creativity":                 "style/awkward",

    # terminology
    "terminology/inappropriate":        "terminology/inappropriate for context",
    "terminology/inconsistent":         "terminology/inconsistent use",

    # locale convention
    "locale convention/date format":    "locale convention/date",
    "date_time_format":                 "locale convention/date",
    "locale_convention/date":           "locale convention/date",
    "locale convention/time format":    "locale convention/time",
    "locale_convention/time":           "locale convention/time",
    "locale convention/currency format":"locale convention/currency",
    "locale_convention/currency":       "locale convention/currency",
    "currency_format":                  "locale convention/currency",
    "locale convention/name format":    "locale convention/name",
    "locale convention/telephone format":"locale convention/telephone",
    "locale convention/address format": "other",
    "number_format":                    "other",
    "measurement_format":               "other",

    # non-translation
    "non-translation!":                 "non-translation",
}


class DataPreparator:
    """Prepare training and evaluation datasets."""
    
    def __init__(
        self,
        data_config: DataConfig,
        seed_config: SeedConfig,
        template_config: PromptTemplateConfig,
        prompt_builder: PromptBuilder,
        source_lang: str,
        target_lang: str
    ):
        self.data_config = data_config
        self.seed_config = seed_config
        self.template_config = template_config
        self.prompt_builder = prompt_builder
        self.source_lang = source_lang
        self.target_lang = target_lang

    
    def load_and_filter_data(self) -> pd.DataFrame:
        """Load and filter JSONL data."""
        data_paths = self.data_config.data_paths
        
        if not data_paths:
            raise ValueError("No data paths provided in config")
        
        data = []
        for data_path in data_paths:
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} entries from {len(data_paths)} file(s)")
        
        # Filter
        initial_count = len(df)

        # Remove entries where the paragraph is the whole document
        df = df[df['para_source'] != df['doc_source']]
        logger.info(
            f"Removed para_source == doc_source: {initial_count} -> {len(df)} "
            f"({initial_count - len(df)} removed)"
        )

        # Normalize error types to canonical categories
        def _normalize_error_type(error_type: str) -> str:
            """Map raw error type strings to canonical prompt categories."""
            return ERROR_TYPE_MAPPING.get(error_type, error_type)

        # Count how many types were actually remapped for logging
        remapped_count = 0
        def _normalize_errors_counted(errors):
            nonlocal remapped_count
            if not isinstance(errors, list):
                return errors
            for e in errors:
                if isinstance(e, dict) and 'category' in e:
                    original = e['category']
                    normalized = _normalize_error_type(original)
                    if normalized != original:
                        remapped_count += 1
                    e['category'] = normalized
            return errors

        df['para_errors'] = df['para_errors'].apply(_normalize_errors_counted)
        logger.info(f"Normalized error types: {remapped_count} remapped to canonical categories")

        # Remove errors with score == 0 from para_errors
        def _filter_zero_score_errors(errors):
            if not isinstance(errors, list):
                return errors
            return [e for e in errors if e.get('score', None) != 0]

        errors_before = df['para_errors'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).sum()
        df['para_errors'] = df['para_errors'].apply(_filter_zero_score_errors)
        errors_after = df['para_errors'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        ).sum()
        logger.info(
            f"Removed zero-score errors: {errors_before} -> {errors_after} "
            f"({errors_before - errors_after} removed)"
        )

        # If reference-based, remove entries where the system is the human reference
        if not self.template_config.is_reference_less:
            before_refa = len(df)
            df = df[df['system'] != 'refA']
            logger.info(
                f"Reference-based mode: removed refA entries: "
                f"{before_refa} -> {len(df)} ({before_refa - len(df)} removed)"
            )
        
        # Keep only rows with non-null para_errors and not excluded domains
        after_para_filter = len(df)
        df = df[
            pd.notnull(df['para_errors']) &
            ~df['domain'].isin(self.data_config.exclude_domains)
        ]

        logger.info(
            f"Filtered (null errors + excluded domains): "
            f"{after_para_filter} -> {len(df)} "
            f"({after_para_filter - len(df)} removed)"
        )
        logger.info(
            f"Total: {initial_count} -> {len(df)} "
            f"({len(df)/initial_count*100:.1f}% retained)"
        )

        # Deduplicate exact base examples only.
        #
        # We use a common comparable key across all settings so that paragraph-level,
        # document-level, reference-less, and reference-based experiments start from
        # the same base records.
        #
        # Important:
        # para_errors is a list of dicts, so it must be serialized into a stable key
        # before using drop_duplicates.
        def _stable_json_key(value) -> str:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)

        before_dedup = len(df)

        df = df.assign(
            _para_errors_key=df['para_errors'].apply(_stable_json_key)
        )

        dedup_cols = [
            'para_source',
            'para_translation',
            '_para_errors_key',
        ]

        df = (
            df
            .drop_duplicates(subset=dedup_cols, keep='first')
            .drop(columns=['_para_errors_key'])
        )

        logger.info(
            f"Deduplicated exact duplicates on common comparable key {dedup_cols}: "
            f"{before_dedup} -> {len(df)} ({before_dedup - len(df)} removed)"
        )

        # Remove records only when the same extracted span is repeated pathologically.
        # Remove only if the same span text appears at least threshold times.
        def _has_pathologically_repeated_spans(
            errors,
            para_source: str,
            para_translation: str,
            threshold: int = 5,
        ) -> bool:
            if not isinstance(errors, list) or len(errors) <= 1:
                return False

            span_counts = {}

            for err in errors:
                if not isinstance(err, dict):
                    continue

                if not (
                    isinstance(err.get('start'), int)
                    and isinstance(err.get('end'), int)
                ):
                    continue

                start, end = err['start'], err['end']
                text = para_source if err.get('is_source_error', False) else para_translation

                # Ignore malformed offsets.
                if start < 0 or end < 0 or start >= end or end > len(text):
                    continue

                span_text = text[start:end]

                # Ignore empty / whitespace-only spans.
                if not span_text.strip():
                    continue

                span_counts[span_text] = span_counts.get(span_text, 0) + 1

                if span_counts[span_text] >= threshold:
                    return True

            return False

        before_repeat = len(df)

        repeated_mask = df.apply(
            lambda row: _has_pathologically_repeated_spans(
                row['para_errors'],
                row['para_source'],
                row['para_translation'],
                threshold=5,
            ),
            axis=1,
        )

        df = df[~repeated_mask]

        logger.info(
            f"Removed records with pathologically repeated error spans "
            f"(same span text >= 5 times): "
            f"{before_repeat} -> {len(df)} ({before_repeat - len(df)} removed)"
        )

        return df
    
    def split_by_source(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split dataset by unique sources to avoid data leakage.
        
        Returns:
            (train_records, eval_records)
        """
        unique_sources = df['para_source'].unique()
        
        train_sources, eval_sources = train_test_split(
            unique_sources,
            test_size=self.data_config.test_size,
            random_state=self.seed_config.seed
        )
        
        train_df = df[df['para_source'].isin(train_sources)]
        eval_df = df[df['para_source'].isin(eval_sources)]
        
        logger.info(f"Training set: {len(train_df)} rows")
        logger.info(f"Evaluation set: {len(eval_df)} rows")
        logger.info(f"Unique sources - Total: {len(unique_sources)}")
        logger.info(f"Train: {len(train_sources)} ({len(train_sources)/len(unique_sources)*100:.1f}%)")
        logger.info(f"Eval: {len(eval_sources)} ({len(eval_sources)/len(unique_sources)*100:.1f}%)")
        
        return train_df.to_dict('records'), eval_df.to_dict('records')
    
    def generate_prompts(self, records: List[Dict]) -> List[str]:
        """
        Generate prompts based on template configuration.
        
        Generates prompts according to template_config settings:
        - If use_all_templates=True: generates all 4 variations
        - Otherwise: generates only the specified combination
        """
        prompts = []
        active_templates = self.template_config.get_active_templates()
        
        logger.info(f"Generating prompts with {len(active_templates)} template(s):")
        for include_doc, is_ref_less in active_templates:
            template_type = (
                f"{'Document' if include_doc else 'Paragraph'}-level, "
                f"{'reference-less' if is_ref_less else 'reference-based'}"
            )
            logger.info(f"  - {template_type}")
        
        for record in records:
            for include_doc, is_ref_less in active_templates:
                if include_doc:
                    prompt = self.prompt_builder.build_document_prompt(
                        self.source_lang,
                        self.target_lang,
                        record['doc_source'],
                        record['doc_translation'],
                        record['para_source'],
                        record['para_reference'],
                        record['para_translation'],
                        record['para_errors'],
                        is_reference_less=is_ref_less
                    )
                else:
                    prompt = self.prompt_builder.build_paragraph_prompt(
                        self.source_lang,
                        self.target_lang,
                        record['para_source'],
                        record['para_reference'],
                        record['para_translation'],
                        record['para_errors'],
                        is_reference_less=is_ref_less
                    )
                
                prompts.append(prompt)
        
        return prompts
    
    def prepare(self, save_path: Path, tokenizer=None) -> Tuple[Dataset, Dataset]:
        """
        Main preparation pipeline.
        
        Returns:
            (train_dataset, eval_dataset)
        """
        # Load and filter
        df = self.load_and_filter_data()
        
        # Split by source
        train_records, eval_records = self.split_by_source(df)
        
        # Generate prompts
        logger.info("Generating training prompts...")
        train_prompts = self.generate_prompts(train_records)
        
        logger.info("Generating evaluation prompts...")
        eval_prompts = self.generate_prompts(eval_records)
            
        # Create datasets
        train_dataset = Dataset.from_list([{"text": t} for t in train_prompts])
        eval_dataset = Dataset.from_list([{"text": t} for t in eval_prompts])
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        # Save datasets to JSON
        self._save_datasets(train_dataset, eval_dataset, save_path)

        # Generate and save dataset summary
        self._save_summary(
            train_records=train_records,
            eval_records=eval_records,
            train_prompts=train_prompts,
            eval_prompts=eval_prompts,
            save_path=save_path,
            tokenizer = tokenizer
        )
        
        return train_dataset, eval_dataset
    
    def _save_datasets(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        save_path: Path
    ):
        """Save datasets to JSON files."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        train_path = save_path / "train_dataset.json"
        eval_path = save_path / "eval_dataset.json"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset.to_dict(), f, ensure_ascii=False)
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_dataset.to_dict(), f, ensure_ascii=False)
        
        logger.info(f"Datasets saved to {save_path}")

    @staticmethod
    def save_combined_datasets(
        train_dataset: Dataset,
        eval_dataset: Dataset,
        save_path: Path,
    ):
        """
        Save the final combined (multi-lp) datasets to JSON files.
        Called from main after concatenate + shuffle.
        """
        save_path.mkdir(parents=True, exist_ok=True)

        train_path = save_path / "train_dataset.json"
        eval_path  = save_path / "eval_dataset.json"

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_dataset.to_dict(), f, ensure_ascii=False)

        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_dataset.to_dict(), f, ensure_ascii=False)

        logger.info(
            f"Combined datasets saved to {save_path} "
            f"({len(train_dataset)} train, {len(eval_dataset)} eval)"
        )

    def _save_summary(
        self,
        train_records: List[Dict],
        eval_records: List[Dict],
        train_prompts: List[str],
        eval_prompts: List[str],
        save_path: Path,
        tokenizer=None
    ):
        """Generate summary statistics and write them to dataset_summary.txt."""
        summary = FinetuningDatasetSummarizer.generate_split_summary(
            train_records=train_records,
            eval_records=eval_records,
            train_prompts=train_prompts,
            eval_prompts=eval_prompts,
            tokenizer=tokenizer
        )
        FinetuningDatasetSummarizer.save_summary(summary, save_path)