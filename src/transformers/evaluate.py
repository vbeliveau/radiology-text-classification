#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:28:21 2023

@author: vbeliveau
"""

import argparse
import json
import os
import numpy as np
import torch
import transformers

from datasets import Dataset
from pathlib import Path
from peft import PeftModel
from sklearn.metrics import (
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


from utils import (
    load_and_preproc_data,
    load_tokenizer,
    root_dir,
    set_logging,
)


def compute_metrics(labels, predictions):
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    f1_score_macro =  f1_score(labels, predictions, average="macro")
    return {
        "balanced_accuracy": balanced_accuracy,
        "f1_score_macro": f1_score_macro,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Training of classifier head for transformer model.')
    parser.add_argument("-c", "--config_json", type=str, required=True)
    args = parser.parse_args()

    # Load configs and assign parameters
    with open(args.config_json, "r") as f:
        configs = json.load(f)

    with open(configs["training_configs"], "r") as f:
        training_args_dict = json.load(f)

    project_name = configs["project_name"]
    base_model = configs["base_model"]
    model_id = training_args_dict["output_dir"]

    # Assign paths
    os.chdir(root_dir)
    output_dir = training_args_dict["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    set_logging()

    # Load config and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load and preprocess data
    data = load_and_preproc_data(project_name, tokenizer)
    eval_tokenized_ds = data["eval_tokenized_ds"]
    n_classes = data["n_classes"]
    classes_weight = data["classes_weight"]

    # Map outputs lists of tensors, need to convert to tensors only
    eval_tokenized_ds.set_format(
        "pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
    eval_dataloader = DataLoader(
        eval_tokenized_ds, collate_fn=data_collator, batch_size=128)

    # Load model
    if configs.get("peft", False):
        print(configs.get("base_model"))
        print(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            configs.get("base_model"), num_labels=n_classes)
        model = PeftModel.from_pretrained(model, model_id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Compute predictions
    model.eval()
    labels = []
    predictions = []

    for batch in eval_dataloader:
        # [print(f"{k}: {type(v)}") for k, v in batch.items()]
        labels += [batch.pop("labels")]
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions += [torch.argmax(logits, dim=-1).cpu().numpy()]

    labels = np.hstack(labels)
    predictions = np.hstack(predictions)

    # Compute F1-score macro
    metrics = compute_metrics(labels, predictions)
    print(metrics)

    # Plot confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(cm)
    # cm_display = ConfusionMatrixDisplay(cm).plot()

    # # Store predictions
    # df_eval = eval_dataset.to_pandas()
    # df_eval["pred_label"] = pred_labels

    # # Identify labels that were not correctly predicted
    # df_mislabeled = None
    # for label in list(set(df_eval["label"])):
    #     df_mislabeled = pd.concat([df_mislabeled, df_eval.query(
    #         "pred_label != label and label == @label")])