"""Data loading utilities for WMT preprocessing."""

import os
import json
import logging
from typing import Dict, Tuple, List, Optional
import re
import pandas as pd

logger = logging.getLogger(__name__)


class MQMAnnotationLoader:
    """Handles loading of MQM annotations."""

    SEVERITY_WEIGHTS = {
        'critical': -10,
        'major': -5,
        'minor': -1,
    }

    @staticmethod
    def _compute_score(mapped_errors) -> float:
        score = 0.0
        if not isinstance(mapped_errors, list):
            return score
        for err in mapped_errors:
            if isinstance(err, dict):
                score += MQMAnnotationLoader.SEVERITY_WEIGHTS.get(
                    err.get('severity', ''), 0
                )
        return score

    @staticmethod
    def load_annotations(mqm_path: str) -> Dict[Tuple[str, int, str], Tuple]:
        """
        Load MQM WMT25 annotations from a JSON file.

        Returns a mapping from (system_id, segment_id, doc_id) to
        (mapped_errors, source_segment, reference_segment, hypothesis_segment, mqm_score).
        """
        if not os.path.exists(mqm_path):
            logger.warning(f"MQM file not found: {mqm_path}")
            return {}

        try:
            with open(mqm_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {mqm_path}: {e}")
            return {}
        except IOError as e:
            logger.error(f"Failed to read file {mqm_path}: {e}")
            return {}

        df = pd.DataFrame(dataset)

        # Validate required columns
        required_cols = [
            "system_id", "segment_id", "source_segment",
            "reference_segment", "hypothesis_segment", "doc_id", "mapped_errors"
        ]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns in MQM file: {missing_cols}")
            return {}

        mqm_mapping = {}
        for _, row in df.iterrows():
            try:
                system_id = str(row["system_id"])
                seg_id = int(row["segment_id"])
                doc_id = str(row["doc_id"])

                key = (system_id, seg_id, doc_id)
                value = (
                    row["mapped_errors"],
                    str(row["source_segment"]),
                    str(row["reference_segment"]),
                    str(row["hypothesis_segment"]),
                    None,
                )

                if key in mqm_mapping:
                    logger.warning(f"Duplicate key found: {key}. Overwriting.")

                mqm_mapping[key] = value

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue

        logger.info(
            f"Loaded MQM annotations for {len(mqm_mapping)} segments "
            f"from {mqm_path}"
        )
        return mqm_mapping
    

class MQMWMT22TSVLoader:

    SEVERITY_WEIGHTS = {
        'critical': -10,
        'major': -5,
        'minor': -1,
    }

    @staticmethod
    def _strip_csv_quoting(text: str) -> str:
        # pandas with sep='\t' sometimes leaves outer CSV quotes un-stripped,
        # returning e.g. '"content with escaped-quotes"' verbatim.
        # Strip the outer " and unescape "" -> " when both conditions hold.
        if text.startswith('"') and text.endswith('"') and '""' in text:
            return text[1:-1].replace('""', '"')
        return text.replace('""', '"')

    @staticmethod
    def _extract_spans(candidate: str) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Strip <v>...</v> tags from candidate and return clean text + span positions.
        Positions refer to the clean text (tags removed).
        """
        clean = ""
        spans = []
        last = 0
        for m in re.finditer(r'<v>(.*?)</v>', candidate):
            clean += candidate[last:m.start()]
            start = len(clean)
            clean += m.group(1)
            spans.append((start, len(clean)))
            last = m.end()
        clean += candidate[last:]
        return clean, spans

    @staticmethod
    def load_annotations(
        tsv_path: str,
        wmt,
    ) -> Dict[Tuple[str, int, str], Tuple]:

        if not os.path.exists(tsv_path):
            logger.warning(f"MQM TSV file not found: {tsv_path}")
            return {}

        df = pd.read_csv(tsv_path, sep="\t")

        # Keep only the lowest-numbered rater for each (doc, seg, system)
        df["_rater_num"] = df["rater_id"].str.extract(r"rater(\d+)").astype(int)
        min_rater = (
            df.groupby(["document_id", "doc_segment_id", "system_id"])["_rater_num"]
            .min()
            .reset_index()
            .rename(columns={"_rater_num": "_min_rater_num"})
        )
        df = df.merge(min_rater, on=["document_id", "doc_segment_id", "system_id"])
        df = df[df["_rater_num"] == df["_min_rater_num"]].copy()
        df.drop(columns=["_rater_num", "_min_rater_num"], inplace=True)

        doc_start = {doc_name: start for doc_name, (start, _) in wmt.docs.items()}

        mqm_mapping = {}

        for (doc_name, local_seg_id, system_id), group in df.groupby(
            ["document_id", "doc_segment_id", "system_id"]
        ):
            if doc_name not in doc_start:
                logger.warning(f"Document '{doc_name}' not found in EvalSet, skipping.")
                continue

            global_seg_id = doc_start[doc_name] + int(local_seg_id) - 1
            first = group.iloc[0]

            errors = []
            mqm_score = 0.0
            clean_candidate = None
            clean_source = None

            for _, row in group.iterrows():
                category = row["category"]
                severity = row["severity"]

                source_has_tag = '<v>' in str(row["source"])
                candidate_has_tag = '<v>' in str(row["candidate"])

                # Determine whether the error is on the source or the hypothesis
                if source_has_tag and not candidate_has_tag:
                    is_source_error = True
                    clean, spans = MQMWMT22TSVLoader._extract_spans(str(row["source"]))
                    if clean_source is None:
                        clean_source = clean
                    if clean_candidate is None:
                        clean_candidate, _ = MQMWMT22TSVLoader._extract_spans(
                            MQMWMT22TSVLoader._strip_csv_quoting(str(row["candidate"]))
                        )

                else:
                    is_source_error = False
                    clean, spans = MQMWMT22TSVLoader._extract_spans(
                        MQMWMT22TSVLoader._strip_csv_quoting(str(row["candidate"]))
                    )
                    if clean_candidate is None:
                        clean_candidate = clean
                    if clean_source is None:
                        clean_source, _ = MQMWMT22TSVLoader._extract_spans(str(row["source"]))

                if category == "no-error":
                    continue

                if spans:
                    start, end = spans[0]
                    if not clean[start:end].strip() or end > len(clean) or start >= end:
                        start, end = None, None
                else:
                    start, end = None, None

                errors.append({
                    "start": start,
                    "end": end,
                    "category": category,
                    "severity": severity,
                    "is_source_error": is_source_error,
                })
                mqm_score += MQMWMT22TSVLoader.SEVERITY_WEIGHTS.get(severity, 0)

            # Fallback in case clean_candidate was never set
            if clean_candidate is None:
                clean_candidate, _ = MQMWMT22TSVLoader._extract_spans(
                    MQMWMT22TSVLoader._strip_csv_quoting(str(first["candidate"]))
                )
            clean_candidate = clean_candidate.strip()

            # Use wmt.src for consistency with doc_source (uniform encoding);
            # fall back to the TSV clean_source if global_seg_id is out of range
            try:
                src_segment = wmt.src[global_seg_id].strip()
            except IndexError:
                logger.warning(
                    f"global_seg_id {global_seg_id} out of range "
                    f"(wmt.src len={len(wmt.src)}) for doc '{doc_name}' "
                    f"local_seg_id={local_seg_id}. Using TSV source as fallback."
                )
                src_segment = (clean_source or str(first["source"])).strip()

            # Use wmt.sys_outputs for consistency with doc_translation (same encoding source)
            try:
                hyp_segment = wmt.sys_outputs[str(system_id)][global_seg_id].strip()
            except (KeyError, IndexError):
                logger.warning(
                    f"sys_output not found for {system_id}[{global_seg_id}], using TSV fallback."
                )
                hyp_segment = clean_candidate

            # Re-validate spans after swapping in src/hyp_segment
            # (offsets were computed on the TSV text, wmt.src may differ in length)
            validated_errors = []
            for err in errors:
                s, e = err.get('start'), err.get('end')
                if s is not None and e is not None:
                    text = src_segment if err.get('is_source_error') else hyp_segment
                    if s < 0 or e > len(text) or s >= e:
                        err = {**err, 'start': None, 'end': None}
                validated_errors.append(err)

            key = (str(system_id), global_seg_id, str(doc_name))
            value = (
                validated_errors,
                src_segment,
                str(first["reference"]).strip(),
                hyp_segment,
                mqm_score,
            )

            if key in mqm_mapping:
                logger.warning(f"Duplicate key {key}, overwriting.")
            mqm_mapping[key] = value


        logger.info(
            f"Loaded WMT22 MQM annotations for {len(mqm_mapping)} segments from {tsv_path}"
        )
        return mqm_mapping