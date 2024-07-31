#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:34:00 2024

@author: vbeliveau
"""

import argparse
import pandas as pd

from pathlib import Path

from utils import (
    eval_few_shots,
    get_model_info,
    load_data,
    load_pipeline,
    predict_samples,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/nlp/data/preproc")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="/nlp/models/llm")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="/nlp")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

# %%

    # Define paths
    model_id, model_str = get_model_info(args.model, args.root_dir)
    out_dir = f"{args.out_dir}/{model_str}/{args.project}"
    metrics_csv = f"{out_dir}/eval_metrics.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not Path(metrics_csv).exists():

        pipe = load_pipeline(model_id)
        data = load_data(args.project, data_dir=args.data_dir, val=True)
        df_train = data["train_dataset"].to_pandas()
        df_eval = data["eval_dataset"].to_pandas()
        unique_labels = df_train["label_text"].unique()

        if args.n_samples is None:
            val_metrics = pd.read_csv(f"{out_dir}/val_metrics.csv")
            n_samples = int(
                val_metrics.iloc[val_metrics["f1_score_macro"].idxmax()]["n_samples"])
            print(f"Optimal n_samples={n_samples}")
        else:
            n_samples = args.n_samples

        pred, status = predict_samples(
            pipe,
            df_train,
            df_eval,
            unique_labels,
            max_retry=5,
            model_id=args.model,
            n_samples=n_samples,
            valid_labels=unique_labels,
            verbose=args.verbose,
        )

        y_pred = pred
        y_true = list(df_eval["label_text"])
        y_pred = [y_pred[i] for i in range(len(pred)) if status[i]]
        y_true = [y_true[i] for i in range(len(pred)) if status[i]]
        metrics = {"n_samples": n_samples}
        metrics = eval_few_shots(y_true, y_pred, unique_labels)

        df_metrics = pd.DataFrame(metrics, index=[0])
        df_results = pd.DataFrame({
            "label_true": list(df_eval["label_text"]),
            "label_pred": pred,
            "status": status
        })

        df_metrics.to_csv(metrics_csv, index=False)
        df_results.to_csv(
            f"{out_dir}/eval_results_n-samples-{n_samples}.csv", index=False)

    else:
        print(f"Output file {metrics_csv} exists. Skipping.")
