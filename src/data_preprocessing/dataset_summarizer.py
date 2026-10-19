"""Dataset summarization utilities."""

import logging
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetSummarizer:
    """Generates summary statistics for the dataset."""
    
    @staticmethod
    def generate_summary(dataset: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive summary of the dataset."""
        if not dataset:
            return "Dataset is empty"
        
        df = pd.DataFrame(dataset)
        
        summary_lines = [
            "\n" + "="*80,
            "DATASET SUMMARY",
            "="*80,
            f"\nTotal entries: {len(df)}",
            f"\nColumns: {list(df.columns)}",
        ]
        
        # Non-null counts for all columns
        summary_lines.append(f"\nNon-null counts by column:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            percentage = (non_null / len(df)) * 100
            summary_lines.append(
                f"  - {col}: {non_null}/{len(df)} ({percentage:.2f}%) "
                f"[{null_count} null]"
            )
        
        # Language pairs
        if 'lang_pair' in df.columns:
            non_null = df['lang_pair'].notna().sum()
            summary_lines.append(f"\nLanguage pairs:")
            for lp, count in df['lang_pair'].value_counts().items():
                summary_lines.append(f"  - {lp}: {count}")
        
        # Systems
        if 'system' in df.columns:
            non_null = df['system'].notna().sum()
            unique = df['system'].nunique()
            summary_lines.append(f"\nNumber of Systems: {unique} unique")
            summary_lines.append(f"System Names:")
            for sys in sorted(set(df['system'])):
                summary_lines.append(f"  - {sys}")
        
        # Domains
        if 'domain' in df.columns:
            non_null = df['domain'].notna().sum()
            summary_lines.append(f"\nDomains:")
            for domain, count in df['domain'].value_counts().items():
                summary_lines.append(f"  - {domain}: {count}")
        
        # Documents
        if 'doc_name' in df.columns:
            non_null = df['doc_name'].notna().sum()
            unique = df['doc_name'].nunique()
            summary_lines.append(f"\nDocuments: {unique} unique")
            
        # Scores
        if 'para_score' in df.columns:
            non_null_scores = df['para_score'].dropna()
            null_count = df['para_score'].isna().sum()
            if len(non_null_scores) > 0:
                summary_lines.append(f"\nParagraph scores:")
                summary_lines.append(f"  - Non-null: {len(non_null_scores)}/{len(df)} ({len(non_null_scores)/len(df)*100:.2f}%)")
                summary_lines.append(f"  - Null: {null_count}")
                summary_lines.append(f"  - Mean: {non_null_scores.mean():.4f}")
                summary_lines.append(f"  - Std: {non_null_scores.std():.4f}")
                summary_lines.append(f"  - Min: {non_null_scores.min():.4f}")
                summary_lines.append(f"  - Max: {non_null_scores.max():.4f}")
            else:
                summary_lines.append(f"\nParagraph scores: All null ({null_count}/{len(df)})")
        
        # Errors
        if 'para_errors' in df.columns:
            non_null_errors = df['para_errors'].notna().sum()
            null_errors = df['para_errors'].isna().sum()
            entries_with_errors = df['para_errors'].apply(
                lambda x: x is not None and len(x) > 0 if isinstance(x, list) else False
            ).sum()
            entries_with_empty_errors = df['para_errors'].apply(
                lambda x: x is not None and len(x) == 0 if isinstance(x, list) else False
            ).sum()
            
            summary_lines.append(f"\nError annotations:")
            summary_lines.append(f"  - Non-null: {non_null_errors}/{len(df)} ({non_null_errors/len(df)*100:.2f}%)")
            summary_lines.append(f"  - Null: {null_errors}")
            summary_lines.append(f"  - With errors (len > 0): {entries_with_errors} ({entries_with_errors/len(df)*100:.2f}%)")
            summary_lines.append(f"  - Empty list (len = 0): {entries_with_empty_errors}")
            
            # Error count statistics
            if entries_with_errors > 0:
                error_counts = df['para_errors'].apply(
                    lambda x: len(x) if isinstance(x, list) and x else 0
                )
                non_zero_error_counts = error_counts[error_counts > 0]
                summary_lines.append(f"  - Avg errors per annotated entry (only len > 0): {non_zero_error_counts.mean():.2f}")
                summary_lines.append(f"  - Avg errors per annotated entry: {error_counts[error_counts.index.isin(df['para_errors'].notna()[df['para_errors'].notna()].index)].mean():.2f}")
                summary_lines.append(f"  - Max errors in one entry: {non_zero_error_counts.max()}")
        
        # Context windows
        if 'doc_indices' in df.columns:
            non_null = df['doc_indices'].notna().sum()
            window_sizes = df['doc_indices'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            non_zero_windows = window_sizes[window_sizes > 0]
            
            if len(non_zero_windows) > 0:
                summary_lines.append(f"\nContext window sizes:")
                summary_lines.append(f"  - Mean: {non_zero_windows.mean():.2f}")
                summary_lines.append(f"  - Median: {non_zero_windows.median():.0f}")
                min_val = non_zero_windows.min()
                summary_lines.append(f"  - Min: {min_val} ({(non_zero_windows == min_val).sum()} entries)")
                max_val = non_zero_windows.max()
                summary_lines.append(f"  - Max: {max_val} ({(non_zero_windows == max_val).sum()} entries)")
                    
        # Text field statistics
        text_fields = ['para_source', 'para_translation', 'para_reference', 
                       'doc_source', 'doc_translation']
        
        for field in text_fields:
            if field in df.columns:
                non_null = df[field].notna().sum()
                null_count = df[field].isna().sum()
                
                # Filter non-null and non-empty strings
                valid_texts = df[field].dropna().astype(str)
                non_empty = valid_texts[valid_texts.str.len() > 0]
                
                summary_lines.append(f"\n{field}:")
                if len(non_empty) > 0:
                    lengths = non_empty.str.len()
                    summary_lines.append(f"  - Length (chars) - Mean: {lengths.mean():.2f}")
                    summary_lines.append(f"  - Length (chars) - Median: {lengths.median():.0f}")
                    summary_lines.append(f"  - Length (chars) - Min: {lengths.min()}")
                    summary_lines.append(f"  - Length (chars) - Max: {lengths.max()}")
        
        # Memory usage
        summary_lines.append(f"\nMemory usage:")
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum() / 1024 / 1024  # Convert to MB
        summary_lines.append(f"  - Total: {total_memory:.2f} MB")
        
        # Top columns by memory
        top_memory = memory_usage.sort_values(ascending=False).head(5)
        summary_lines.append(f"  - Top 5 columns by memory:")
        for col, mem in top_memory.items():
            summary_lines.append(f"    * {col}: {mem / 1024 / 1024:.2f} MB")
        
        summary_lines.append("\n" + "="*80 + "\n")
        
        return "\n".join(summary_lines)
    
    @staticmethod
    def save_summary_to_file(dataset: List[Dict[str, Any]], output_path: str):
        """Generate and save summary to a text file."""
        summary = DatasetSummarizer.generate_summary(dataset)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"Summary saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save summary to {output_path}: {e}")
            raise