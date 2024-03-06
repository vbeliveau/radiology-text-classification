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

from pathlib import Path
from setfit import (
    # sample_dataset,
    SetFitModel,
    Trainer,
    TrainingArguments,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm, trange
from typing import Optional, List, Union
from utils import (
    default_batch_size,
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

    print(configs)
    batch_size = configs.get("batch_size", default_batch_size)
    body_learning_rate = configs["body_learning_rate"]
    head_learning_rate = configs["head_learning_rate"]
    l2_weight = configs["l2_weight"]
    num_epochs = configs["num_epochs"]
    project_name = configs["project_name"]

    # Assign paths
    os.chdir(root_dir)
    output_dir = f"{root_dir}/models/setfit/{project_name}/end_to_end"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(project_name)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    n_classes = data["n_classes"]
    classes_weight = data["classes_weight"]

    total_train_examples = 25000

    args = TrainingArguments(
        batch_size=batch_size,
        body_learning_rate=body_learning_rate,
        end_to_end=True,
        eval_steps=1,
        evaluation_strategy="epoch",
        head_learning_rate=head_learning_rate,
        l2_weight=l2_weight,
        logging_dir=f"{output_dir}/runs",
        logging_steps=1,
        logging_strategy="steps",
        max_length=128,
        max_steps=num_epochs,
        num_epochs=num_epochs,
        output_dir=f"{output_dir}/checkpoints",
        report_to=["tensorboard"],
        sampling_strategy="oversampling",
        save_strategy="epoch",
        save_steps=1,
        save_total_limit=1,
        use_amp=True,
    )

    model_body = SentenceTransformer(
        f"{root_dir}/models/setfit/{project_name}/model_body/best_model")

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
