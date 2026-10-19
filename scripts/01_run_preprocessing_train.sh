#!/bin/bash

# Builds the WMT22/WMT23 training data used for fine-tuning: each segment is
# preprocessed together with its full source and target document.

# ============================================================
# Configuration
# ============================================================
# Use the entire document as context for doc_source/doc_translation.
ENABLE_CONTEXT_WINDOWS="no"

# Dataset-language pairs used as fine-tuning training data (WMT22/WMT23)
PAIRS=(
    "wmt22:en-de"
    "wmt22:en-ru"
    "wmt22:en-zh"
    "wmt22:zh-en"
    "wmt23:en-de"
    "wmt23:he-en"
    "wmt23:zh-en"
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

echo "All train preprocessing runs completed."
