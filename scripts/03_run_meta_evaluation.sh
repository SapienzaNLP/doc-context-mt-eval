#!/bin/bash

# ============================================================
# Configuration (override via env vars instead of editing this file), for example:
#   MODEL_NAME=Qwen3-32B MODEL_PATH=Qwen/Qwen3-32B bash scripts/03_run_meta_evaluation.sh
#   ENABLE_THINKING=yes PAIRS_RAW=wmt24:en-de bash scripts/03_run_meta_evaluation.sh
# ============================================================
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
ENABLE_THINKING="${ENABLE_THINKING:-no}"
ARE_AGGREGATED_RESULTS="${ARE_AGGREGATED_RESULTS:-no}"
WANDB_NEW_RUN="${WANDB_NEW_RUN:-yes}"
WANDB_MODE="${WANDB_MODE:-offline}"

# Derived from ENABLE_THINKING, matching how 02_run_evaluation.sh names its runs:
# meta-evaluation has to look up the exact wandb_run_name that produced the
# responses CSV it reads, so the two must stay in sync. (main_meta_evaluation.py
# has no --decoding flag -- passing one errors out.)
DECODING=$([ "$ENABLE_THINKING" = "yes" ] && echo "thinking" || echo "greedy")

# Dataset-language pairs: comma-separated "dataset:lp" list, override via PAIRS_RAW
PAIRS_RAW="${PAIRS_RAW:-wmt24:en-de,wmt24:en-es,wmt24:ja-zh,wmt25:en-ko_KR,wmt25:ja-zh_CN}"
IFS=',' read -ra PAIRS <<< "$PAIRS_RAW"

# ============================================================
# Combinations: (is_reference_less, include_document_context)
# ============================================================
declare -A COMBOS
COMBOS["yes:no"]="para-base_ref-less"
COMBOS["no:no"]="para-base_ref-based"
COMBOS["yes:yes"]="para-doc_ref-less"
COMBOS["no:yes"]="para-doc_ref-based"

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

        RUN_NAME="${RUN_PREFIX}_${MODEL_NAME}_${LP}_${DECODING}"
        PROJECT_NAME="doc-context-eval-${LP}_${DATASET}"

        echo "============================================================"
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

echo "All meta-evaluation runs completed."
