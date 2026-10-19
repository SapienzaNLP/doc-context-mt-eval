#!/bin/bash

# Meta-evaluates the XCOMET-XL baseline against the WMT MQM gold annotations,
# for comparison with the LLM scenarios in 03_run_meta_evaluation.sh.

# ============================================================
# Configuration (override via env vars instead of editing this file), for example:
#   PAIRS_RAW=wmt24:en-de bash scripts/03_run_meta_evaluation_baselines.sh
# ============================================================
MODEL_NAME="${MODEL_NAME:-XCOMET-XL}"
MODEL_PATH="${MODEL_PATH:-Unbabel/XCOMET-XL}"
ARE_AGGREGATED_RESULTS="${ARE_AGGREGATED_RESULTS:-no}"
WANDB_NEW_RUN="${WANDB_NEW_RUN:-yes}"
WANDB_MODE="${WANDB_MODE:-online}"

# Dataset-language pairs: comma-separated "dataset:lp" list, override via PAIRS_RAW
PAIRS_RAW="${PAIRS_RAW:-wmt24:en-de,wmt24:en-es,wmt24:ja-zh}"
IFS=',' read -ra PAIRS <<< "$PAIRS_RAW"

# ============================================================
# Combinations: (is_reference_less, include_document_context)
# ============================================================
# XCOMET is reference-less/reference-based only; it has no document-context
# variant, unlike the LLM evaluation scenarios.
declare -A COMBOS
COMBOS["yes:no"]="para-base_ref-less"
COMBOS["no:no"]="para-base_ref-based"

# ============================================================
# Run loop
# ============================================================
for pair in "${PAIRS[@]}"; do
    DATASET="${pair%%:*}"
    LP="${pair##*:}"

    for combo_key in "${!COMBOS[@]}"; do
        REF_LESS="${combo_key%%:*}"
        INCLUDE_DOC="${combo_key##*:}"
        RUN_PREFIX="${COMBOS[$combo_key]}"

        RUN_NAME="xcomet_xl_${RUN_PREFIX}_${LP}"
        PROJECT_NAME="doc-context-eval-${LP}_${DATASET}"

        echo "============================================================"
        echo "Model:      $MODEL_NAME"
        echo "Dataset:    $DATASET"
        echo "LP:         $LP"
        echo "Ref-less:   $REF_LESS  |  Include doc: $INCLUDE_DOC"
        echo "Run name:   $RUN_NAME"
        echo "Project:    $PROJECT_NAME"
        echo "------------------------------------------------------------"

        python src/meta_evaluation/main_meta_evaluation.py \
            --dataset "$DATASET" \
            --lp "$LP" \
            --is_reference_less "$REF_LESS" \
            --include_document_context "$INCLUDE_DOC" \
            --wandb_run_name "$RUN_NAME" \
            --wandb_project_name "$PROJECT_NAME" \
            --are_aggregated_results "$ARE_AGGREGATED_RESULTS" \
            --wandb_new_run "$WANDB_NEW_RUN" \
            --model_name "$MODEL_PATH" \
            --wandb_mode "$WANDB_MODE"

        echo ""
    done
done

echo "All baseline meta-evaluation runs completed."
