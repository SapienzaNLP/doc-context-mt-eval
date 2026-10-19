"""Main script for preprocessing WMT evaluation datasets."""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from mt_metrics_eval import data

from config_preprocessing import ProcessingConfig
from loaders import MQMAnnotationLoader, MQMWMT22TSVLoader
from builders import DocumentIndexBuilder
from merger import DatasetMerger
from dataset_summarizer import DatasetSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mono_context_windows(n_paragraphs: float, step: int = 1) -> list[tuple[int,int]]:
    """Generate mono-lateral context windows (0,X) and (X,0)."""
    total = round(n_paragraphs)
    configs = []
    for window_size in range(0, total + 1, step):
        if window_size == 0:
            configs.append((0, 0))  # 0p0f -> no context
        else:
            configs.append((0, window_size))  # 0pXf
            configs.append((window_size, 0))  # Xp0f
    return configs


class WMTDataProcessor:
    """Main processor for WMT datasets."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process_all(self) -> List[Dict[str, Any]]:
        """Process all language pairs and return aggregated dataset."""
        aggregated_dataset = []
        for lp in self.config.language_pairs:
            logger.info(f"Processing language pair: {lp}")
            try:
                dataset = self.process_language_pair(lp)
                aggregated_dataset.extend(dataset)
                logger.info(f"Processed {len(dataset)} entries for {lp}")
            except Exception as e:
                logger.error(f"Failed to process {lp}: {e}", exc_info=True)
                continue
        return aggregated_dataset

    def process_language_pair(self, lp: str) -> List[Dict[str, Any]]:
        """Process a single language pair and return dataset entries."""
        wmt = data.EvalSet(
            self.config.dataset,
            lp,
            read_stored_metric_scores=True,
            read_stored_ratings=True
        )

        if self.config.enable_context_windows:
            logger.info(
                f"Context windows enabled: {self.config.n_pre} preceding + "
                f"{self.config.n_post} following segments"
            )
        else:
            logger.info("Context windows disabled: using entire document as context")

        index_to_doc = DocumentIndexBuilder.build(wmt)
        mqm_mapping = self._load_mqm_mapping(lp, wmt)

        merger = DatasetMerger(wmt, index_to_doc, mqm_mapping, self.config)
        return merger.merge()

    def _load_mqm_mapping(self, lp: str, wmt) -> Optional[Dict]:
        if self.config.dataset == 'wmt25':
            mqm_path = f"data-wmt/original/mqm_generalMT2025_{lp}_with_errors.json"
            return MQMAnnotationLoader.load_annotations(mqm_path)
        elif self.config.dataset == 'wmt22' and lp == 'en-zh':
            mqm_path = "data-wmt/original/mqm_generalMT2022_enzh.tsv"
            return MQMWMT22TSVLoader.load_annotations(mqm_path, wmt)
        return None

    def save_results(self, dataset: List[Dict[str, Any]]):
        output_path = Path("data-wmt/preprocessed/mqm") / self.config.dataset / self.config.output_file

        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"Saved {len(dataset)} entries to {output_path}")



def parse_arguments() -> ProcessingConfig:
    parser = argparse.ArgumentParser(
        description="Preprocess WMT evaluation datasets with MQM annotations"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process (e.g., wmt24)."
    )

    parser.add_argument(
        "--lp",
        type=str,
        required=True,
        help="Comma-separated list of language pairs (e.g., en-de,en-fr)."
    )

    parser.add_argument(
        "--enable_context_windows",
        type=str,
        choices=["yes", "no"],
        default="no",
        help="Whether to enable document-level context windows. "
            "If 'yes', n_pre and/or n_post must be specified."
    )

    parser.add_argument(
        "--n_pre",
        type=int,
        default=None,
        help="Number of preceding sentences to include as context. "
            "Used only if --enable_context_windows=yes."
    )

    parser.add_argument(
        "--n_post",
        type=int,
        default=None,
        help="Number of following sentences to include as context. "
            "Used only if --enable_context_windows=yes."
    )

    parser.add_argument(
        "--max_context",
        type=int,
        default=None,
        help="If set, automatically generates asymmetric context windows "
            "(0,X) and (X,0) for X in [1, max_context]. "
            "Overrides n_pre and n_post."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file where the processed dataset will be saved."
    )

    args = parser.parse_args()

    lp_list = [lp.strip() for lp in args.lp.split(",")]
    return ProcessingConfig(
        dataset=args.dataset,
        language_pairs=lp_list,
        enable_context_windows=(args.enable_context_windows=="yes"),
        n_pre=args.n_pre,
        n_post=args.n_post,
        max_context=args.max_context,
        output_file=args.output_file,
    ), args.max_context


def main():
    try:
        config, max_context = parse_arguments()
        print("Configuration:")
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")

        if max_context is not None:
            output_dir = Path("context_windows")
            output_dir.mkdir(parents=True, exist_ok=True)

            window_sizes = generate_mono_context_windows(max_context)
            print("context windows:", window_sizes)
            for n_pre, n_post in window_sizes:
                logger.info(f"Running context window: n_pre={n_pre}, n_post={n_post}")

                output_file_name = f"{Path(config.output_file).stem}_{n_pre}p{n_post}f.jsonl"
                window_config = ProcessingConfig(
                    dataset=config.dataset,
                    language_pairs=config.language_pairs,
                    enable_context_windows=config.enable_context_windows,
                    n_pre=n_pre,
                    n_post=n_post,
                    max_context=max_context,
                    output_file=str(output_dir / output_file_name)
                )
                processor = WMTDataProcessor(window_config)
                aggregated_dataset = processor.process_all()

                summary = DatasetSummarizer.generate_summary(aggregated_dataset)
                logger.info(summary)

                processor.save_results(aggregated_dataset)
                
                summary_file_name = f"{Path(window_config.output_file).stem}_summary.txt"
                summary_path = Path("data-wmt/preprocessed/mqm") / config.dataset / window_config.output_file
                summary_path = summary_path.parent / summary_file_name 
                summary_path.parent.mkdir(parents=True, exist_ok=True)
    
                DatasetSummarizer.save_summary_to_file(aggregated_dataset, output_path=summary_path)   
        else:
            processor = WMTDataProcessor(config)
            aggregated_dataset = processor.process_all()

            summary = DatasetSummarizer.generate_summary(aggregated_dataset)
            logger.info(summary)

            processor.save_results(aggregated_dataset)
            DatasetSummarizer.save_summary_to_file(aggregated_dataset, output_path=Path("data-wmt/preprocessed/mqm") / config.dataset / f"{Path(config.output_file).stem}_summary.txt")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
