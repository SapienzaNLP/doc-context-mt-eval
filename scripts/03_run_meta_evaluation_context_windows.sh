#!/bin/bash

# ============================================================
# Configuration (override via env vars instead of editing this file), for example:
#   DATASET=wmt24 LP=en-de bash scripts/03_run_meta_evaluation_context_windows.sh
# ============================================================
MODEL_NAME="${MODEL_NAME:-Qwen3-32B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
DATASET="${DATASET:-wmt25}"
LP="${LP:-ja-zh_CN}"
MAX_CONTEXT="${MAX_CONTEXT:-20}"
ARE_AGGREGATED_RESULTS="${ARE_AGGREGATED_RESULTS:-no}"
WANDB_NEW_RUN="${WANDB_NEW_RUN:-yes}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ============================================================
# Context windows: 0pNf (N=0..MAX_CONTEXT) and Np0f (N=0..MAX_CONTEXT)
# Matches the sweep produced by 01_run_preprocessing_context_windows.sh.
# ============================================================
CONTEXT_WINDOWS=()
for n in $(seq 0 "$MAX_CONTEXT"); do
    CONTEXT_WINDOWS+=("0p${n}f")
done
for n in $(seq 0 "$MAX_CONTEXT"); do
    CONTEXT_WINDOWS+=("${n}p0f")
done

# ============================================================
# Run loop
# ============================================================
for CW in "${CONTEXT_WINDOWS[@]}"; do

    RUN_PREFIX="para-doc_ref-less"
    RUN_NAME="${RUN_PREFIX}_${MODEL_NAME}_${LP}_${CW}"
    PROJECT_NAME="doc-context-eval-${LP}_${DATASET}_context_windows"

    echo "============================================================"
    echo "Dataset:    $DATASET"
    echo "LP:         $LP"
    echo "Context window: $CW"
    echo "Run name:   $RUN_NAME"
    echo "Project:    $PROJECT_NAME"
    echo "------------------------------------------------------------"

    python src/meta_evaluation/main_meta_evaluation.py \
        --dataset "$DATASET" \
        --lp "$LP" \
        --is_reference_less "yes" \
        --include_document_context "yes" \
        --wandb_run_name "$RUN_NAME" \
        --wandb_project_name "$PROJECT_NAME" \
        --are_aggregated_results "$ARE_AGGREGATED_RESULTS" \
        --wandb_new_run "$WANDB_NEW_RUN" \
        --model_name "$MODEL_PATH" \
        --wandb_mode "$WANDB_MODE"

    echo ""
done

echo "All context window meta-evaluation runs completed."
