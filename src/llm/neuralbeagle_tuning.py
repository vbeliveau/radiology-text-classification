#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:34:00 2024

@author: vbeliveau
"""

import argparse
import pandas as pd
import torch

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from utils import (
    eval_few_shots,
    load_data,
    predict_samples,
)


# %%


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Evaluate neuralbeagle classification.")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--min_samples", type=int, required=True, default=1)
    parser.add_argument("--max_samples", type=int, required=True, default=1)
    parser.add_argument("--data_dir", type=str, required=True)    
    args = parser.parse_args()

    out_dir = f"/nlp/models/llm/munin-neuralbeagle-7b/{args.project}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    model_id = "RJuro/munin-neuralbeagle-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # load_in_4bit=True,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )

    # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    # model.generation_config.pad_token_ids = tokenizer.pad_token_id

    pipe = pipeline(
        "text-generation",
        device_map="auto",
        model=model,
        # pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )

    # data = load_data(
    #     args.project, data_dir="/proc_data1/bd5/nlp/data/preproc-midl", val=True)
    data = load_data(
        args.project, data_dir=args.data_dir, val=True)
    df_train = data["train_dataset"].to_pandas()
    df_val = data["val_dataset"].to_pandas()
    df_eval = data["eval_dataset"].to_pandas()
    unique_labels = df_train["label_text"].unique()
    
    metrics_csv = f"{out_dir}/val_metrics.csv"

    if Path(metrics_csv).is_file():
        df_metrics = pd.read_csv(metrics_csv)
    else:
        df_metrics = None

    for n_samples in range(args.min_samples, args.max_samples + 1):

        if df_metrics is None or n_samples not in df_metrics["n_samples"]:

            print(f"Processing n_samples={n_samples}")

            pred, status = predict_samples(
                pipe,
                df_train,
                df_val,
                unique_labels,
                max_retry=5,
                model_id="munin",
                n_samples=n_samples,
                valid_labels=unique_labels,
                # verbose=True,
            )

            y_pred = pred
            y_true = list(df_val["label_text"])
            y_pred = [y_pred[i] for i in range(len(pred)) if status[i]]
            y_true = [y_true[i] for i in range(len(pred)) if status[i]]
            metrics = {"n_samples": n_samples}
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
                f"{out_dir}/val_results_n-samples-{n_samples}.csv", index=False)
