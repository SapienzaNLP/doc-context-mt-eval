<div align="center">

<h1>
    Exploring the Impact of Document Context on<br>
    LLM-based Machine Translation Evaluation
</h1>

[![Paper](https://img.shields.io/badge/EMNLP%202026%20Paper-red)](https://aclanthology.org/2026.findings-emnlp.1839/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MT Metrics Eval](https://shields.io/badge/-MT%20Metrics%20Eval-green?style=flat&logo=github&labelColor=gray)](https://github.com/google-research/mt-metrics-eval)

</div>

This work investigates the impact of document-level context on LLM-based Machine Translation evaluation, considering the availability of human references, thinking mode, and task-specific fine-tuning. Results show that the benefits of document context vary across datasets and may partly stem from input repetition. Fine-tuning consistently improves evaluation performance, with larger gains when document context is available.

This repository contains the code for the paper *Exploring the Impact of Document Context on LLM-based Machine Translation Evaluation*, accepted at Findings of EMNLP 2026.

## Repository structure

```
doc-context-mt-eval/
├── src/
│   ├── data_preprocessing/   # WMT loading, MQM annotations, document context
│   ├── evaluation/           # Prompts, model inference, error-span extraction
│   ├── meta_evaluation/      # Span matching and meta-evaluation metrics
│   └── finetuning/           # Training-data preparation and LoRA fine-tuning
├── scripts/                  # Preprocessing, evaluation, meta-evaluation, fine-tuning drivers
├── pyproject.toml
└── .gitignore
```

## Setup

```bash
pip install -e .
```

Python 3.9+. Core dependencies are version-pinned in `pyproject.toml` to the exact versions used to produce the results in this repo. Install from a fresh environment to reproduce them closely.

## External dependencies

This pipeline builds on top of two external repositories:

- [`mt-metrics-eval`](https://github.com/google-research/mt-metrics-eval): WMT metrics meta-evaluation toolkit, also used to fetch and cache the WMT data.

```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
```

For an exact reproduction, record the revision of `mt-metrics-eval` used for data preparation.

- [`span-mt-metrics-eval`](https://github.com/sted19/span-mt-metrics-eval): implements the MPP (Match with Partial overlap and Partial credit) span-matching metric used in meta-evaluation. Installs automatically via `pip install -e .`, no manual clone needed.

## Pipeline

1. **Preprocessing**: `src/data_preprocessing/` turns raw WMT MQM data into JSONL, grouping segments by document and attaching each one's surrounding context, driven by `scripts/01_run_preprocessing_*.sh`.
2. **Evaluation**: `src/evaluation/` runs LLM-based MT error detection across the [evaluation scenarios](#evaluation-scenarios) below, driven by `scripts/02_run_evaluation.sh`.
3. **Meta-evaluation**: `src/meta_evaluation/` scores predicted error spans against human judgments using a severity-weighted variant of MPP (Match with Partial overlap and Partial credit), driven by `scripts/03_run_meta_evaluation*.sh`.
4. **Fine-tuning**: `src/finetuning/` fine-tunes models (LoRA) separately for each of the [evaluation scenarios](#evaluation-scenarios) on MQM-style error detection data, driven by `scripts/04_run_finetuning.sh`.

## Data

`data-wmt/` holds the WMT data and is expected as a sibling directory next to `src/`; it is not shipped in this repo. Raw data is fetched and cached automatically by `mt-metrics-eval` on first use.

```bash
bash scripts/01_run_preprocessing_eval.sh   # WMT24 + WMT25 -> test data for evaluation
bash scripts/01_run_preprocessing_train.sh  # WMT22 + WMT23 -> training data for fine-tuning
```

Preprocessed JSONL files are written to `data-wmt/preprocessed/mqm/<dataset>/preprocessed_<dataset>_<lp>.jsonl`, each example containing the target translation segment and the document-level information needed by the evaluation configurations.

To build the data for the context-window experiment from Section 6 of the paper (see [Evaluation scenarios](#evaluation-scenarios) for how to run it):

```bash
bash scripts/01_run_preprocessing_context_windows.sh
```

## Evaluation scenarios

To study the impact of document-level context on LLM-based MT evaluation and its interaction with the availability of manually curated references, `src/evaluation/main_evaluation.py` (and the fine-tuning pipeline) evaluate each translation segment under four configurations, crossing two axes: whether the model sees the full source/target document or only the isolated segment (DL vs. SL), and whether it has access to a human reference translation or must rely on the source alone (RB vs. RF):

| Scenario | `--include_document_context` | `--is_reference_less` |
|---|---|---|
| SL-RF (segment, reference-free)   | no  | yes |
| SL-RB (segment, reference-based)  | no  | no  |
| DL-RF (document, reference-free)  | yes | yes |
| DL-RB (document, reference-based) | yes | no  |

`--repeat_segment yes` (reference-free, segment-level only) runs the SL-RF<sub>rep</sub> repetition control, which duplicates the segment instead of showing the full document, used to disentangle the effect of document context from simply repeating the input.

`scripts/02_run_evaluation.sh` loops over all four scenarios (plus the SL-RF<sub>rep</sub> control) for a given model and set of language pairs; model, pairs, and other settings are overridable via environment variables instead of editing the script. For example:

```bash
MODEL_NAME=google/gemma-4-E4B-it ENABLE_THINKING=yes PAIRS_RAW=wmt24:en-de bash scripts/02_run_evaluation.sh
```

Responses and run settings are saved under `outputs/metrics_results/<dataset>/<lp>/<model_name>/<run_name>/predicted_score/` (`all_responses.csv` and `run_config.txt`); `all_responses.csv` contains the model responses and predicted error spans consumed by the meta-evaluation stage.

**Context-window experiment.** Once `scripts/01_run_preprocessing_context_windows.sh` has built the sweep data (up to 20 additional segments by default), evaluate each window size by pointing `main_evaluation.py` at its file directly with `--data_path`, instead of letting it resolve the input from `--dataset`/`--lp`:

```bash
python src/evaluation/main_evaluation.py \
    --data_path data-wmt/preprocessed/mqm/wmt24/context_windows/preprocessed_wmt24_en-de_sweep_2p0f.jsonl \
    --dataset wmt24 \
    --lp en-de \
    --model_name Qwen/Qwen3-32B \
    --is_reference_less yes \
    --include_document_context yes \
    --repeat_segment no \
    --enable_thinking no \
    --wandb_run_name para-doc_ref-less_Qwen3-32B_en-de_2p0f \
    --wandb_project_name doc-context-eval-en-de_wmt24_context_windows \
    --wandb_mode offline
```

The matching meta-evaluation driver is `scripts/03_run_meta_evaluation_context_windows.sh`. Its `MODEL_NAME`, `MODEL_PATH`, `DATASET`, `LP`, and `MAX_CONTEXT` settings must match the evaluation runs.

## Meta-evaluation

`src/meta_evaluation/` scores the error spans produced by evaluation against the gold human annotations, computing the **F1 MPP** (Match with Partial overlap and Partial credit) via [`span-mt-metrics-eval`](https://github.com/sted19/span-mt-metrics-eval), with optimal bipartite matching and severity weights `minor=0.2`, `major=1`, `critical=1`.

```bash
MODEL_NAME=Qwen3-32B MODEL_PATH=Qwen/Qwen3-32B PAIRS_RAW=wmt24:en-de bash scripts/03_run_meta_evaluation.sh
```

Additional drivers cover aggregation across languages (`03_run_meta_evaluation_all_languages.sh`), baseline metrics (`03_run_meta_evaluation_baselines.sh`), and the context-window experiment (`03_run_meta_evaluation_context_windows.sh`). These expect the corresponding metric outputs to have been generated already.

## Fine-tuning

`src/finetuning/` LoRA fine-tunes `Qwen/Qwen3-8B` separately for each of the four [evaluation scenarios](#evaluation-scenarios) on MQM-style error detection data.

```bash
bash scripts/04_run_finetuning.sh
```

By default, the driver prepares the training data before launching fine-tuning (set `RUN_PREPROCESSING=no` to reuse already preprocessed data) and writes model outputs under `outputs/finetuned/<run_name>`. It uses the settings in `src/finetuning/accelerate_config.yaml`; check the configured number of processes and GPU availability before running.

## Quick start

Preprocess the test data (all WMT24/WMT25 language pairs):

```bash
bash scripts/01_run_preprocessing_eval.sh
```

To preprocess a single language pair instead, e.g. WMT24 English–German, call the preprocessing entry point directly:

```bash
python src/data_preprocessing/main_preprocessing.py \
    --dataset wmt24 \
    --lp en-de \
    --enable_context_windows no \
    --output_file preprocessed_wmt24_en-de.jsonl
```

Run the four [evaluation scenarios](#evaluation-scenarios) plus the SL-RF<sub>rep</sub> repetition control for WMT24 English–German with Qwen3-32B:

```bash
MODEL_NAME=Qwen/Qwen3-32B PAIRS_RAW=wmt24:en-de bash scripts/02_run_evaluation.sh
```

Then run [meta-evaluation](#meta-evaluation) against human annotations:

```bash
MODEL_NAME=Qwen3-32B MODEL_PATH=Qwen/Qwen3-32B PAIRS_RAW=wmt24:en-de bash scripts/03_run_meta_evaluation.sh
```

Keep the model identifier, language pairs, and `ENABLE_THINKING` setting consistent between evaluation and meta-evaluation: the latter locates responses using the run names generated by the former.

## Cite this work

This work has been accepted at Findings of EMNLP 2026. If you use this code or build on our findings, please consider citing our paper as follows:

```bibtex
@inproceedings{aur-etal-2026-exploring,
    title = "Exploring the Impact of Document Context on {LLM}-based Machine Translation Evaluation",
    author = "Aur, Marina Iuliana  and
      Perrella, Stefano  and
      Navigli, Roberto",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.1839/",
    pages = "39253--39276",
    ISBN = "979-8-89176-462-0"
}
```

## License

MIT. See [LICENSE](LICENSE).
