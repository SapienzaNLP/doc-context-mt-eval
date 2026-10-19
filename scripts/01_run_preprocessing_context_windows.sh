#!/bin/bash

# Builds the data for the paper's Section 6 analysis ("When Is Document
# Context Useful?"): for each number of additional context segments N from
# 1 to MAX_CONTEXT, one JSONL with N preceding segments and one with N
# following segments, plus the no-context baseline at N=0.

# ============================================================
# Configuration
# ============================================================
MAX_CONTEXT=20

# Same test sets used for the main evaluation (WMT24 + WMT25)
PAIRS=(
    "wmt24:en-de"
    "wmt24:en-es"
    "wmt24:ja-zh"
    "wmt25:en-ko_KR"
    "wmt25:ja-zh_CN"
)

# ============================================================
# Run loop
# ============================================================
for pair in "${PAIRS[@]}"; do
    DATASET="${pair%%:*}"
    LP="${pair##*:}"
    OUTPUT_FILE="preprocessed_${DATASET}_${LP}_sweep.jsonl"

    echo "============================================================"
    echo "Dataset:      $DATASET"
    echo "LP:           $LP"
    echo "Max context:  $MAX_CONTEXT"
    echo "------------------------------------------------------------"

    python src/data_preprocessing/main_preprocessing.py \
        --dataset "$DATASET" \
        --lp "$LP" \
        --enable_context_windows yes \
        --max_context "$MAX_CONTEXT" \
        --output_file "$OUTPUT_FILE"
    echo ""
done

echo "All context-window sweep runs completed."
