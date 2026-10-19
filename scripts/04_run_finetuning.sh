#!/bin/bash

# Fine-tunes Qwen/Qwen3-8B (LoRA) separately for each of the 4 evaluation scenarios
# (paragraph/document-level x reference-based/reference-less), combining all
# configured WMT22/WMT23 language pairs into a single training set per scenario.

# ============================================================
# Configuration (override via env vars instead of editing this file), for example:
#   RUN_PREPROCESSING=no PAIRS_RAW=wmt22:en-de,wmt23:zh-en bash scripts/04_run_finetuning.sh
# ============================================================
MODEL_NAME="Qwen/Qwen3-8B"
MODEL_SHORT_NAME="${MODEL_NAME##*/}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-src/finetuning/accelerate_config.yaml}"
RUN_PREPROCESSING="${RUN_PREPROCESSING:-yes}"
EXCLUDE_DOMAINS="${EXCLUDE_DOMAINS:-speech}"
TEST_SIZE="${TEST_SIZE:-0.1}"
WANDB_PROJECT="${WANDB_PROJECT:-doc-context-finetune}"
WANDB_MODE="${WANDB_MODE:-online}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-outputs/finetuned}"
SEED="${SEED:-42}"

# Training/LoRA hyperparameters (defaults match paper Table 9). BATCH_SIZE is
# per-GPU; with accelerate_config.yaml's num_processes=4 this gives an effective
# batch size of 32.
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Dataset-language pairs used as fine-tuning training data, override via PAIRS_RAW
PAIRS_RAW="${PAIRS_RAW:-wmt22:en-de,wmt22:en-ru,wmt22:en-zh,wmt22:zh-en,wmt23:en-de,wmt23:zh-en}"
IFS=',' read -ra PAIRS <<< "$PAIRS_RAW"

# ============================================================
# Preprocessing: build the per-LP training JSONL files if missing
# ============================================================
if [ "$RUN_PREPROCESSING" = "yes" ]; then
    echo "============================================================"
    echo "Running preprocessing for fine-tuning training data..."
    echo "------------------------------------------------------------"
    bash scripts/01_run_preprocessing_train.sh
    echo ""
fi

DATA_PATHS=""
LPS=""
for pair in "${PAIRS[@]}"; do
    DATASET="${pair%%:*}"
    LP="${pair##*:}"
    DATA_PATH="data-wmt/preprocessed/mqm/${DATASET}/preprocessed_${DATASET}_${LP}.jsonl"

    if [ ! -f "$DATA_PATH" ]; then
        echo "Missing $DATA_PATH, skipping $pair"
        continue
    fi

    DATA_PATHS="${DATA_PATHS:+${DATA_PATHS},}${DATA_PATH}"
    LPS="${LPS:+${LPS},}${LP}"
done

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
for combo_key in "${!COMBOS[@]}"; do
    REF_LESS="${combo_key%%:*}"
    INCLUDE_DOC="${combo_key##*:}"
    RUN_PREFIX="${COMBOS[$combo_key]}"
    RUN_NAME="${RUN_PREFIX}_${MODEL_SHORT_NAME}"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"

    echo "============================================================"
    echo "Scenario:   $RUN_PREFIX"
    echo "LPs:        $LPS"
    echo "Model:      $MODEL_NAME"
    echo "Run name:   $RUN_NAME"
    echo "Output dir: $OUTPUT_DIR"
    echo "------------------------------------------------------------"

    accelerate launch --config_file "$ACCELERATE_CONFIG" src/finetuning/main_finetuning.py \
        --model_name "$MODEL_NAME" \
        --data_paths "$DATA_PATHS" \
        --lp "$LPS" \
        --exclude_domains "$EXCLUDE_DOMAINS" \
        --test_size "$TEST_SIZE" \
        --use_single_template \
        --include_document_context "$INCLUDE_DOC" \
        --is_reference_less "$REF_LESS" \
        --learning_rate "$LEARNING_RATE" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --max_length "$MAX_LENGTH" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --output_dir "$OUTPUT_DIR" \
        --wandb_project_name "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME" \
        --wandb_mode "$WANDB_MODE" \
        --seed "$SEED"

    echo ""
done

echo "All fine-tuning runs completed."
