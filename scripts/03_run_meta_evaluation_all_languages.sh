#!/usr/bin/env bash

set -euo pipefail

# Configuration (override via env vars instead of editing this file), for example:
#   DATASET=wmt24 LPS_RAW=en-de,en-es,ja-zh MODEL_NAME=Qwen3-32B bash scripts/03_run_meta_evaluation_all_languages.sh

BASE_DIR="outputs/metrics_results"
DATASET="${DATASET:-wmt25}"

# Display name used in run names, vendor/folder are the two path components
# under {dataset}/{lp}/ for per-LP results (for example Qwen/Qwen3-8B,
# google/gemma-4-E4B-it, Unbabel/XCOMET-XL) -- note the aggregated output
# path below uses MODEL_NAME alone, not vendor/folder.
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
VENDOR="${VENDOR:-Qwen}"
FOLDER="${FOLDER:-Qwen3-8B}"

LPS_RAW="${LPS_RAW:-en-ko_KR,ja-zh_CN}"
IFS=',' read -ra LPS <<< "$LPS_RAW"

DECODINGS=("greedy" "thinking")
REFS=("ref-less" "ref-based")
PARAS=("para-base" "para-doc")

# suffix appended to per-LP and aggregated run folder names; set to "" for no suffix
SUFFIX="${SUFFIX:-}"

WANDB_PROJECT="${WANDB_PROJECT:-doc-context-eval_${DATASET}_last_and_least}"
WANDB_MODE="${WANDB_MODE:-offline}"

SCRIPT="src/meta_evaluation/main_meta_evaluation.py"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] No files will be written or commands executed."
fi

# Helpers

ref_flag() {
    case "$1" in
        ref-less)  echo "yes" ;;
        ref-based) echo "no"  ;;
    esac
}

para_flag() {
    case "$1" in
        para-base) echo "no"  ;;
        para-doc)  echo "yes" ;;
    esac
}

# Main loop

for DEC in "${DECODINGS[@]}"; do
for REF in "${REFS[@]}"; do
for PARA in "${PARAS[@]}"; do

    # aggregated output uses MODEL_NAME so main_meta_evaluation.py finds the CSV
    RUN_NAME="${PARA}_${REF}_${MODEL_NAME}_all_${DEC}${SUFFIX}"
    OUT_DIR="${BASE_DIR}/${DATASET}/aggregated_results_mqm/${MODEL_NAME}/${RUN_NAME}/predicted_score"
    OUT_CSV="${OUT_DIR}/all_responses.csv"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Combo: dec=${DEC}  ref=${REF}  para=${PARA}"
    echo "  Output CSV : ${OUT_CSV}"
    echo "  WandB run  : ${RUN_NAME}"

    # Step 1: build explicit paths for every LP

    LP_FILES=()
    LP_TAGS=()
    for LP in "${LPS[@]}"; do
        # per-LP folder pattern: {PARA}_{REF}_{MODEL_NAME}_{LP}_{DEC}{SUFFIX}
        LP_RUN_NAME="${PARA}_${REF}_${MODEL_NAME}_${LP}_${DEC}${SUFFIX}"
        CANDIDATE="${BASE_DIR}/${DATASET}/${LP}/${VENDOR}/${FOLDER}/${LP_RUN_NAME}/predicted_score/all_responses.csv"

        if [[ -f "${CANDIDATE}" ]]; then
            LP_FILES+=("${CANDIDATE}")
            LP_TAGS+=("${LP}")
            echo "  [OK]   LP=${LP} → ${CANDIDATE}"
        else
            echo "  [MISS] LP=${LP}: file not found, skipping."
            echo "         Expected: ${CANDIDATE}"
        fi
    done

    if [[ ${#LP_FILES[@]} -eq 0 ]]; then
        echo "  [SKIP] No input files found for this combo, skipping."
        continue
    fi

    # Step 2: concatenate CSVs (header from first file only)

    if $DRY_RUN; then
        echo "  [DRY] Would create: ${OUT_CSV}"
        echo "  [DRY] Would concatenate ${#LP_FILES[@]} files"
    else
        mkdir -p "${OUT_DIR}"
        # Use Python to concatenate: awk breaks on multi-line quoted CSV fields
        TMPPY=$(mktemp --suffix=.py)
        cat > "$TMPPY" << 'PYEOF'
import csv, sys

csv.field_size_limit(sys.maxsize)

out   = sys.argv[1]
n     = (len(sys.argv) - 2) // 2
files = sys.argv[2:2 + n]
lps   = sys.argv[2 + n:]

# Build unified fieldnames preserving order across all files
all_fields, seen = [], set()
for f in files:
    with open(f, newline="", encoding="utf-8") as fin:
        for col in next(csv.reader(fin)):
            if col not in seen:
                all_fields.append(col)
                seen.add(col)
if "lang_pair" not in seen:
    all_fields = ["lang_pair"] + all_fields

with open(out, "w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=all_fields, extrasaction="ignore")
    writer.writeheader()
    for f, lp in zip(files, lps):
        with open(f, newline="", encoding="utf-8") as fin:
            for row in csv.DictReader(fin):
                if "lang_pair" not in row or not row["lang_pair"]:
                    row["lang_pair"] = lp
                writer.writerow(row)
PYEOF
        python3 "$TMPPY" "${OUT_CSV}" "${LP_FILES[@]}" "${LP_TAGS[@]}"
        rm -f "$TMPPY"
        NROWS=$(python3 -c "
import csv, sys
csv.field_size_limit(sys.maxsize)
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    print(sum(1 for _ in csv.reader(f)) - 1)
" "${OUT_CSV}")
        echo "  [OK]   Concatenated ${#LP_FILES[@]} files → ${NROWS} rows"
    fi

    # Step 3: launch meta-evaluation

    CMD=(
        python "${SCRIPT}"
        --dataset                  "${DATASET}"
        --model_name               "${MODEL_NAME}"
        --lp                       "all"
        --is_reference_less        "$(ref_flag "${REF}")"
        --include_document_context "$(para_flag "${PARA}")"
        --are_aggregated_results   "yes"
        --wandb_run_name           "${RUN_NAME}"
        --wandb_project_name       "${WANDB_PROJECT}"
        --wandb_mode               "${WANDB_MODE}"
        --wandb_new_run            "yes"
    )

    echo "  [CMD]  ${CMD[*]}"

    if $DRY_RUN; then
        echo "  [DRY] Command not executed."
    else
        "${CMD[@]}"
        echo "  [DONE] ${RUN_NAME}"
    fi

done
done
done

echo ""
echo "All aggregated runs complete."
