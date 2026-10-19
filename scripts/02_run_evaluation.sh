#!/bin/bash

# ============================================================
# Configuration (override via env vars instead of editing this file), for example:
#   MODEL_NAME=google/gemma-4-E4B-it ENABLE_THINKING=yes bash scripts/02_run_evaluation.sh
#   PAIRS_RAW=wmt24:en-de bash scripts/02_run_evaluation.sh
#   INCLUDE_REPEAT=no bash scripts/02_run_evaluation.sh   # skip the SL-RFrep control
# ============================================================
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"
MODEL_SHORT_NAME="${MODEL_NAME##*/}"
ENABLE_THINKING="${ENABLE_THINKING:-no}"
EXCLUDE_DOMAINS="${EXCLUDE_DOMAINS:-speech}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
WANDB_MODE="${WANDB_MODE:-online}"
INCLUDE_REPEAT="${INCLUDE_REPEAT:-yes}"

# Derived from ENABLE_THINKING (not an independent variable): responses_generator.py
# always samples with the thinking params when enable_thinking is on and always
# decodes greedily otherwise, so the run name can't drift out of sync with it.
DECODING=$([ "$ENABLE_THINKING" = "yes" ] && echo "thinking" || echo "greedy")

# Dataset-language pairs: comma-separated "dataset:lp" list, override via PAIRS_RAW
PAIRS_RAW="${PAIRS_RAW:-wmt24:en-de,wmt24:en-es,wmt24:ja-zh,wmt25:en-ko_KR,wmt25:ja-zh_CN}"
IFS=',' read -ra PAIRS <<< "$PAIRS_RAW"

# ============================================================
# Combinations: (is_reference_less, include_document_context)
# ============================================================
# ref-less=yes, doc=no  → para-base_ref-less
# ref-less=no,  doc=no  → para-base_ref-based
# ref-less=yes, doc=yes → para-doc_ref-less
# ref-less=no,  doc=yes → para-doc_ref-based
#
# The SL-RFrep repetition control (--repeat_segment yes) is run separately
# below for each pair, since it's ref-less/segment-level only and
# incompatible with --include_document_context yes.

declare -A COMBOS
COMBOS["yes:no"]="para-base_ref-less"
COMBOS["no:no"]="para-base_ref-based"
COMBOS["yes:yes"]="para-doc_ref-less"
COMBOS["no:yes"]="para-doc_ref-based"

run_eval() {
    local dataset="$1" lp="$2" ref_less="$3" include_doc="$4" repeat="$5"
    local run_name="$6" project_name="$7"

    echo "============================================================"
    echo "Dataset:    $dataset"
    echo "LP:         $lp"
    echo "Ref-less:   $ref_less  |  Include doc: $include_doc  |  Repeat: $repeat"
    echo "Run name:   $run_name"
    echo "Project:    $project_name"
    echo "------------------------------------------------------------"

    python src/evaluation/main_evaluation.py \
        --dataset "$dataset" \
        --lp "$lp" \
        --is_reference_less "$ref_less" \
        --batch_size "$BATCH_SIZE" \
        --include_document_context "$include_doc" \
        --repeat_segment "$repeat" \
        --enable_thinking "$ENABLE_THINKING" \
        --exclude_domains "$EXCLUDE_DOMAINS" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --wandb_run_name "$run_name" \
        --wandb_project_name "$project_name" \
        --wandb_mode "$WANDB_MODE" \
        --model_name "$MODEL_NAME"
    echo ""
}

# ============================================================
# Run loop
# ============================================================
for pair in "${PAIRS[@]}"; do
    DATASET="${pair%%:*}"
    LP="${pair##*:}"
    PROJECT_NAME="doc-context-eval-${LP}_${DATASET}"

    for combo_key in "${!COMBOS[@]}"; do
        REF_LESS="${combo_key%%:*}"
        INCLUDE_DOC="${combo_key##*:}"
        RUN_PREFIX="${COMBOS[$combo_key]}"
        RUN_NAME="${RUN_PREFIX}_${MODEL_SHORT_NAME}_${LP}_${DECODING}"

        run_eval "$DATASET" "$LP" "$REF_LESS" "$INCLUDE_DOC" "no" "$RUN_NAME" "$PROJECT_NAME"
    done

    if [ "$INCLUDE_REPEAT" = "yes" ]; then
        RUN_NAME="para-repeat_ref-less_${MODEL_SHORT_NAME}_${LP}_${DECODING}"
        run_eval "$DATASET" "$LP" "yes" "no" "yes" "$RUN_NAME" "$PROJECT_NAME"
    fi
done

echo "All runs completed."
