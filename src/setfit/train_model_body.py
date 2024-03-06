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
from setfit import (
    # sample_dataset,
    SetFitModel,
    Trainer,
    TrainingArguments,
)
from utils import (
    default_batch_size,
    load_data,
    root_dir
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Fine-tuining of SetFit model body.')
    parser.add_argument("-c", "--config_json", type=str, required=True)
    args = parser.parse_args()

    # Load configs and assign parameters
    with open(args.config_json, "r") as f:
        configs = json.load(f)
    project_name = configs["project_name"]
    body_learning_rate = configs["body_learning_rate"]
    batch_size = configs.get("batch_size", default_batch_size)
    model_prefix = f"body-learning-rate-{body_learning_rate}_batch-size-{batch_size}"
    print(model_prefix)

    # Assign paths
    os.chdir(root_dir)
    output_dir = f"{root_dir}/models/setfit/{project_name}/model_body"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(project_name)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]

    total_train_examples = 25000
    total_log_steps = 10

    # Load config
    training_args = TrainingArguments(
        batch_size=batch_size,
        body_learning_rate=body_learning_rate,
        evaluation_strategy="steps",
        eval_steps=total_train_examples/total_log_steps//batch_size,
        logging_dir=f"{output_dir}/runs",
        logging_steps=1,
        max_length=128,
        max_steps=total_train_examples//batch_size,
        num_epochs=1,
        output_dir=f"{output_dir}/checkpoints",
        report_to=["tensorboard"],
        sampling_strategy="oversampling",
        save_strategy="steps",
        save_steps=total_train_examples/total_log_steps//batch_size,
        use_amp=True,
    )

    model = SetFitModel.from_pretrained(
        "sentence-transformers/distiluse-base-multilingual-cased-v2")

    # There is a bug when using containers. Somehow config gets added
    # as an attribute, but is dict instead of PretrainedConfig.
    # Trainer tries to use the to_json_string() mehtod on dict, which
    # throws and error.
    if hasattr(model, "config"):
        delattr(model, "config")

    trainer = Trainer(
        args=training_args,
        metric="accuracy",
        model=model,
    )

    trainer.train_embeddings(
        x_train=train_dataset["text"],
        y_train=train_dataset["label"],
        x_eval=val_dataset["text"],
        y_eval=val_dataset["label"],
    )

    # Save out model
    model_path = f"{output_dir}/best_model"
    trainer.model.save_pretrained(model_path)
