#!/bin/bash

# Builds the WMT24/WMT25 test data used for evaluation: each segment is
# preprocessed together with its full source and target document.

# ============================================================
# Configuration
# ============================================================
# Use the entire document as context for doc_source/doc_translation.
ENABLE_CONTEXT_WINDOWS="no"

# Dataset-language pairs used as test sets for evaluation (WMT24/WMT25)
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
    OUTPUT_FILE="preprocessed_${DATASET}_${LP}.jsonl"

    echo "============================================================"
    echo "Dataset:      $DATASET"
    echo "LP:           $LP"
    echo "Output file:  $OUTPUT_FILE"
    echo "------------------------------------------------------------"

    python src/data_preprocessing/main_preprocessing.py \
        --dataset "$DATASET" \
        --lp "$LP" \
        --enable_context_windows "$ENABLE_CONTEXT_WINDOWS" \
        --output_file "$OUTPUT_FILE"
    echo ""
done

echo "All eval preprocessing runs completed."
