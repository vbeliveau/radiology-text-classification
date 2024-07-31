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
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from utils import (
    load_data,
    root_dir,
    WeightedCELossSetFitHead,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Fine-tuining of SetFit model body.')
    parser.add_argument("-c", "--config_json", type=str, required=True)
    args = parser.parse_args()

    # Load configs and assign parameters
    with open(args.config_json, "r") as f:
        configs = json.load(f)
    project_name = configs["project_name"]
    model_id = configs.get("model_id")
    model_str = configs.get("model_str", model_id.split("/")[-1])

    best_configs_json = f"{root_dir}/models/setfit/{model_str}/{project_name}/head/optuna/best_params.json"
    with open(best_configs_json, "r") as f:
        best_configs = json.load(f)

    print(f"Configs: {configs}")
    print(f"Best configs: {best_configs}")

    batch_size = best_configs.get("batch_size", 128)
    head_learning_rate = best_configs["head_learning_rate"]
    num_epochs = best_configs.get("num_epochs", 50)
    project_name = best_configs["project_name"]

    # Assign paths
    os.chdir(root_dir)
    output_dir = f"{root_dir}/models/setfit/{model_str}/{project_name}/head"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = configs.get("data_dir", "/nlp/data/preproc")
    data = load_data(project_name, data_dir=data_dir)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    n_classes = data["n_classes"]
    classes_weight = data["classes_weight"]

    args = TrainingArguments(
        batch_size=batch_size,
        end_to_end=True,
        eval_steps=1,
        evaluation_strategy="epoch",
        head_learning_rate=head_learning_rate,
        logging_dir=f"{output_dir}/runs",
        logging_steps=1,
        logging_strategy="steps",
        max_length=128,
        num_epochs=num_epochs,
        output_dir=f"{output_dir}/checkpoints",
        report_to=["tensorboard"],
        sampling_strategy="oversampling",
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        use_amp=True,
    )

    model_body = SentenceTransformer(model_id)

    model_head = WeightedCELossSetFitHead(
        in_features=model_body.get_sentence_embedding_dimension(),
        out_features=n_classes,
        weight=classes_weight,
    )

    model_body.to(model_head.device)

    model = SetFitModel(model_body, model_head)

    trainer = Trainer(
        args=args,
        metric=lambda x, y: f1_score(x, y, average="macro"),
        eval_dataset=val_dataset,
        model=model,

    )

    # Train the model end to end
    trainer.train_classifier(
        x_train=train_dataset["text"],
        y_train=train_dataset["label"],
    )
    print(trainer.evaluate())

    # Save out model
    model_path = f"{output_dir}/best_model"
    trainer.model.save_pretrained(model_path)
