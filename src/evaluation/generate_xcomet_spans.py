"""Generate XCOMET-XL span-level predictions for MT evaluation."""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_xcomet_samples(df: pd.DataFrame, is_reference_less: bool) -> list[dict]:
    samples = []

    for row in df.itertuples():
        sample = {
            "src": row.para_source,
            "mt": row.para_translation,
        }

        if not is_reference_less:
            sample["ref"] = row.para_reference

        samples.append(sample)

    return samples


def run_xcomet(
    model,
    dataset: str,
    lp: str,
    df: pd.DataFrame,
    is_reference_less: bool,
    batch_size: int,
    gpus: int,
    output_path: str,
) -> None:
    data = build_xcomet_samples(df, is_reference_less)

    model_output = model.predict(
        data,
        batch_size=batch_size,
        gpus=gpus,
    )

    scores = model_output.scores

    if hasattr(model_output, "metadata") and hasattr(model_output.metadata, "error_spans"):
        error_spans = model_output.metadata.error_spans
    else:
        error_spans = [[] for _ in scores]

    df_out = pd.DataFrame(
        {
            "lang_pair": lp,
            "system": df["system"].values,
            "source": df["para_source"].values,
            "reference": df["para_reference"].values,
            "translation": df["para_translation"].values,
            "gold_errors": df["para_errors"].values,
            "para_score": df["para_score"].values,
            "response": error_spans,
            "comet_score": scores,
            "thinking_content": None,
            "doc_name": df["doc_name"].values,
            "doc_source": df["doc_source"].values,
            "doc_translation": df["doc_translation"].values,
            "para_id": df["para_id"].values,
            "doc_indices": df["doc_indices"].values,
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"Saved {len(df_out)} rows to {output_path}")
    print(f"System-level XCOMET score: {model_output.system_score}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate XCOMET-XL span-level predictions"
    )

    parser.add_argument("--model_name", type=str, default="Unbabel/XCOMET-XL")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lp", type=str, required=True, help="Comma-separated LPs or single LP")
    parser.add_argument("--is_reference_less", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--folder_name", type=str, required=True)

    args = parser.parse_args()

    is_reference_less = args.is_reference_less == "yes"
    lp_list = [lp.strip() for lp in args.lp.split(",")]

    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)

    for lp in lp_list:
        print(f"\nProcessing: {lp}")

        if len(lp_list) == 1:
            dataset_path = (
                f"data-wmt/preprocessed/mqm/{args.dataset}/"
                f"preprocessed_{args.dataset}_{lp}.jsonl"
            )

            output_path = (
                f"outputs/metrics_results/"
                f"{args.dataset}/{lp}/{args.model_name}/{args.folder_name}/"
                f"predicted_score/all_responses.csv"
            )

        else:
            dataset_path = (
                f"data-wmt/preprocessed/mqm/{args.dataset}/"
                f"preprocessed_{args.dataset}_all.jsonl"
            )

            output_path = (
                f"outputs/metrics_results/"
                f"{args.dataset}/aggregated_results_mqm/{args.model_name}/"
                f"{args.folder_name}/predicted_score/all_responses.csv"
            )

        rows = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        df = pd.DataFrame(rows)

        df_filtered = df[
            df["para_errors"].notnull()
            & (df["system"] != "refA")
            & (df["domain"] != "speech")
        ].copy()

        print(f"  Filtered: {len(df)} -> {len(df_filtered)} rows")

        run_xcomet(
            model=model,
            dataset=args.dataset,
            lp=lp,
            df=df_filtered,
            is_reference_less=is_reference_less,
            batch_size=args.batch_size,
            gpus=args.gpus,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()