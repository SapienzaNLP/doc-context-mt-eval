"""Builder classes for document indices and context windows."""

import logging
from typing import List, Dict, Any

from mt_metrics_eval import data

logger = logging.getLogger(__name__)


class DocumentIndexBuilder:
    """Builds index-to-document mapping."""
    
    @staticmethod
    def build(wmt: data.EvalSet) -> Dict[int, Dict[str, Any]]:
        """
        Build mapping from segment index to document metadata.
        
        Each index maps to: doc_name, doc_indices (start, end), 
        doc_source, doc_reference, and domain.
        """
        index_to_doc = {}
        
        # Precompute domain for each index
        index_to_domain = {}
        for domain_name, intervals in wmt.domains.items():
            for start, end in intervals:
                for i in range(start, end):
                    if i in index_to_domain:
                        logger.warning(
                            f"Index {i} already assigned to domain "
                            f"{index_to_domain[i]}, overwriting with {domain_name}"
                        )
                    index_to_domain[i] = domain_name

        # Build document metadata
        for doc_name, (start, end) in wmt.docs.items():
            if start >= end:
                logger.warning(
                    f"Invalid document range for {doc_name}: [{start}, {end})"
                )
                continue
            
            # Extract document segments safely
            try:
                doc_src = [seg.strip() for seg in wmt.src[start:end]]
                doc_ref = [
                    seg.strip()
                    for seg in wmt.all_refs[wmt.std_ref][start:end]
                ]
            except (IndexError, KeyError) as e:
                logger.error(
                    f"Error extracting segments for {doc_name}: {e}"
                )
                continue

            for i in range(start, end):
                index_to_doc[i] = {
                    "doc_name": doc_name,
                    "doc_indices": (start, end),
                    "doc_source": doc_src,
                    "doc_reference": doc_ref,
                    "domain": index_to_domain.get(i, "unknown"),
                }

        logger.info(f"Built index for {len(index_to_doc)} segments")
        return index_to_doc


class ContextWindowBuilder:
    """Builds context windows for segments."""
    
    @staticmethod
    def get_window_indices(
        i: int,
        start: int,
        end: int,
        n_pre: int,
        n_post: int
    ) -> List[int]:
        """
        Get indices for context window around segment i.
        
        Returns indices [i-n_pre, ..., i-1, i, i+1, ..., i+n_post]
        clipped to [start, end).
        """
        if not (start <= i < end):
            logger.warning(
                f"Index {i} out of range [{start}, {end}). "
                "Returning single index."
            )
            return [i]
        
        indices = []
        
        # Add preceding segments
        for k in range(n_pre, 0, -1):
            j = i - k
            if j >= start:
                indices.append(j)
        
        # Add current segment
        indices.append(i)
        
        # Add following segments
        for k in range(1, n_post + 1):
            j = i + k
            if j < end:
                indices.append(j)
        
        return indices