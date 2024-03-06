#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:28:21 2023

@author: vbeliveau
"""


import argparse
import json
import os

from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
)
from utils import (
    compute_metrics,
    load_and_preproc_data,
    load_tokenizer,
    root_dir,
    set_logging,
    WeightedCELossTrainer,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Training of classifier head for transformer model.')
    parser.add_argument("-c", "--config_json", type=str, required=True)
    args = parser.parse_args()

    # Load configs and assign parameters
    with open(args.config_json, "r") as f:
        configs = json.load(f)

    with open(configs["training_configs"], "r") as f:
        training_args_dict = json.load(f)

    project_name = configs["project_name"]
    base_model = configs["base_model"]
    model_id = configs.get("finetuned_model_path", base_model)
    peft = configs.get("peft", False)

    # Assign paths
    os.chdir(root_dir)
    output_dir = training_args_dict["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    set_logging()

    # Load config and tokenizer
    print(f"model_id: {model_id}")
    tokenizer = load_tokenizer(model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load and preprocess data
    data = load_and_preproc_data(project_name, tokenizer)
    train_tokenized_ds = data["train_tokenized_ds"]
    val_tokenized_ds = data["val_tokenized_ds"]
    n_classes = data["n_classes"]
    classes_weight = data["classes_weight"]
    print(f"classes_weight: {classes_weight}")

    # Assign best parameters
    with open(f"{training_args_dict['output_dir']}/optuna/best_params.json") as f:
        optuna_best_params = json.load(f)
    training_args_dict["learning_rate"] = optuna_best_params.get(
        "learning_rate")
    training_args_dict["weight_decay"] = optuna_best_params.get("weight_decay")

    training_parser = HfArgumentParser(TrainingArguments)
    training_args, = training_parser.parse_dict(training_args_dict)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=n_classes)

    trainer = WeightedCELossTrainer(
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        eval_dataset=val_tokenized_ds,
        model=model,
        train_dataset=train_tokenized_ds,
    )
    trainer.classes_weight = classes_weight

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
