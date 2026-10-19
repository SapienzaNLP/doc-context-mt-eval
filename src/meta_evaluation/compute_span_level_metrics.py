from collections import Counter
from typing import List, Dict, Any
import ast

from utils_meta_evaluation import normalize_text, find_error_positions_with_severity, parse_json


def compute_span_level_metrics(
    model_name: str,
    hypothesis_response: str,
    translation: str,
    target_errors: List[Dict],
    span_stats=None,
) -> Dict[str, Any]:
    """
    Parse predicted and gold error spans into the (start, end, severity) format
    required by span-mt-metrics-eval's MPP computation (the paper's reported metric).

    Args:
        model_name: Evaluated model/metric name; selects the parser (COMET-style
            spans with character offsets, vs. the LLM JSON error format located
            by string search).
        hypothesis_response: Raw model/metric output containing predicted errors.
        translation: System translation segment, used to locate predicted spans
            by string search when the model returns span text rather than offsets.
        target_errors: Gold error annotations for this segment.
        span_stats: Optional SpanMatchingStats accumulator for ambiguous-match
            diagnostics (see find_error_positions_with_severity).

    Returns:
        Dict with:
            - check_inconsistency (bool): True if hypothesis_response could not be
              parsed or contained invalid data; such segments contribute no
              predicted spans.
            - smme_pred_spans / smme_gold_spans (List[Dict]): predicted/gold spans
              as {start, end, side, severity} dicts, in span-mt-metrics-eval's
              expected format.
    """
    check_inconsistency = False

    translation_str = translation if isinstance(translation, str) else ""
    normalized_translation = normalize_text(translation_str)

    if "comet" in model_name.lower():
        try:
            errors_list = ast.literal_eval(hypothesis_response)
            hypothesis_offsets = [
                (int(err["start"]), int(err["end"]), err.get("severity"))
                for err in errors_list
            ]
        except Exception:
            hypothesis_offsets = []
            check_inconsistency = True

    else:
        try:
            if model_name == "CohereLabs/c4ai-command-a-03-2025":
                hypothesis_response = hypothesis_response.replace("```json", "").replace("```", "").strip()

            parsed = parse_json(hypothesis_response)
            errors_list = parsed.get("errors", [])
            if not isinstance(errors_list, list):
                check_inconsistency = True
                errors_list = []
            else:
                errors_list = [err for err in errors_list if isinstance(err, dict)]

        except Exception:
            check_inconsistency = True
            errors_list = []

        # Detect repetition loops before the (comparatively expensive) text search below
        if not check_inconsistency and len(errors_list) > 5:
            span_counts = Counter(
                err.get("span") or err.get("minimal_span", "")
                for err in errors_list
            )
            if span_counts.most_common(1)[0][1] > 5:
                check_inconsistency = True
                errors_list = []

        valid_errors = []
        for err in errors_list:
            try:
                severity = err.get("severity")
                if severity is None or severity.lower() not in {"critical", "major", "minor"}:
                    check_inconsistency = True
                valid_errors.append(err)
            except Exception:
                check_inconsistency = True

        hypothesis_offsets = find_error_positions_with_severity(
            normalized_translation, valid_errors, stats=span_stats
        )

    if check_inconsistency:
        return {"check_inconsistency": True, "smme_pred_spans": [], "smme_gold_spans": []}

    # Gold spans: exclude zero-score, source-side, and WMT25's 'missing' sentinel.
    # Both WMT24 and WMT25 apply the same filters.
    target_offsets = [
        (err["start"], err["end"], err.get("severity"))
        for err in target_errors
        if "start" in err and "end" in err
        and err["start"] != "missing" and err["end"] != "missing"
        and err.get("score", 1) != 0
        and err.get("is_source_error") != True
    ]

    return {
        "check_inconsistency": False,
        "smme_pred_spans": [
            {
                "start": int(s), "end": int(e), "side": "target",
                "severity": str(sev).lower() if sev is not None and str(sev).strip() != "" else "undecided",
            }
            for s, e, sev in hypothesis_offsets if int(e) > int(s)
        ],
        "smme_gold_spans": [
            {
                "start": int(s), "end": int(e), "side": "target",
                "severity": str(sev).lower() if sev is not None and str(sev).strip() != "" else "undecided",
            }
            for s, e, sev in target_offsets if int(e) > int(s)
        ],
    }
