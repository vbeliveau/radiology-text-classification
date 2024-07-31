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


class DebugArgs():
    def __init__(self):
        self.data_dir = "/proc_data1/bd5/nlp/data/preproc-melba-translated"
        # self.model = "biomistral"
        self.model = "llama3-70B"
        self.n_samples = "15"
        self.out_dir = "/proc_data1/bd5/nlp/models/llm"
        self.project = "FCD"
        self.root_dir = "/proc_data1/bd5/nlp"
        self.verbose = True


args = DebugArgs()

# %%


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/nlp/data/preproc")
    parser.add_argument("-min_samples", "--min_samples",
                        type=int, required=True, default=1)
    parser.add_argument("-max_samples", "--max_samples",
                        type=int, required=True, default=1)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="/nlp/models/llm")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="/nlp")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

# %%

    model_id, model_str = get_model_info(args.model, args.root_dir)
    out_dir = f"{args.out_dir}/{model_str}/{args.project}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(model_id)

    data = load_data(args.project, data_dir=args.data_dir, val=True)
    df_train = data["train_dataset"].to_pandas()
    df_val = data["val_dataset"].to_pandas()
    unique_labels = df_train["label_text"].unique()

    # %% Test prediction

    # df_ = df_val.iloc[120]
    # pred, status = few_shots_predict(
    #     pipe,
    #     df_["text"],
    #     df_train,
    #     n_samples=5,
    #     valid_labels=unique_labels,
    #     verbose=True
    # )

    # print(
    #     f"True category: {df_['label_text']}, Predicted category: {pred}")

    # %%

    metrics_csv = f"{out_dir}/val_metrics.csv"

    if Path(metrics_csv).is_file():
        df_metrics = pd.read_csv(metrics_csv)
    else:
        df_metrics = None

    if args.n_samples is None:
        print(
            f"Using n_samples in range [{args.min_sampels}, {args.max_samples}]")
        n_samples = range(args.min_samples, args.max_samples + 1)
    else:
        print(f"Using n_samples in {args.n_samples}")
        n_samples = [int(n_samples_)
                     for n_samples_ in args.n_samples.split(",")]

    for n_samples_ in n_samples:

        if df_metrics is None or n_samples_ not in list(df_metrics["n_samples"]):

            print(f"Processing n_samples={n_samples_}")

            pred, status = predict_samples(
                pipe,
                df_train,
                df_val,
                unique_labels,
                max_retry=5,
                model_id=args.model,
                n_samples=n_samples_,
                valid_labels=unique_labels,
                # verbose=True,
            )

            y_pred = pred
            y_true = list(df_val["label_text"])
            y_pred = [y_pred[i] for i in range(len(pred)) if status[i]]
            y_true = [y_true[i] for i in range(len(pred)) if status[i]]
            metrics = {"n_samples": n_samples_}
            metrics.update(
                eval_few_shots(y_true, y_pred, unique_labels)
            )

            df_metrics = pd.concat([
                df_metrics, pd.DataFrame(metrics, index=[0])])
            df_results = pd.DataFrame({
                "label_true": list(df_val["label_text"]),
                "label_pred": pred,
                "status": status
            })

            df_metrics.to_csv(metrics_csv, index=False)
            df_results.to_csv(
                f"{out_dir}/val_results_n-samples-{n_samples_}.csv", index=False)
