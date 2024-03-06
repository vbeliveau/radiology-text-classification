#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:47:01 2024

@author: vbeliveau
"""

import numpy as np
import pandas as pd
import torch
import transformers

from datasets import Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoTokenizer,
    Trainer,
)
from typing import Dict, List, Optional, Tuple, Union


root_dir = '/nlp'


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    f1_score_macro =  f1_score(labels, predictions, average="macro")
    return {
        "balanced_accuracy": balanced_accuracy,
        "f1_score_macro": f1_score_macro,
    }


def load_data(project_name: str):

    df = pd.read_csv(
        f"{root_dir}/data/preproc/{project_name}.csv")
    # Remove nans.. should not be happening at this stage
    df = df[[isinstance(text, str) for text in df["text"]]]

    # Convert labels to numerical categories
    factors = pd.factorize(df['label'])
    categories_summary = "Categories: "

    n_factors = []
    for n_factor, factor in enumerate(factors[1]):
        n_factors += [sum([factor_ == n_factor for factor_ in factors[0]])]
        categories_summary += f" {factor} (n={n_factors[-1]})"
    print(categories_summary)
    df['label'] = factors[0]
    n_classes = len(factors[1])

    def get_min_labels(labels):
        n_labels = [
            len([label_ for label_ in labels if label_ == label])
            for label in labels
        ]
        return min(n_labels)

    # Create train/test datasets
    train_test_dataset = Dataset.from_pandas(
        df).train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_dataset["train"]
    eval_dataset = train_test_dataset["test"]

    # Further split data into train/validation
    df_train = pd.DataFrame(
        {"text": train_dataset["text"], "label": train_dataset["label"]})
    train_val_dataset = Dataset.from_pandas(
        df_train).train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]

    # Count unique classes and get their weight
    classes_count = [len([label_ for label_ in train_dataset["label"] if label_ == label])
                     for label in list(set(train_dataset["label"]))]
    print(f"Classes count in training dataset: {classes_count}")
    classes_weight = Tensor([1/count for count in classes_count])

    return {
        "classes_count": classes_count,
        "classes_weight": classes_weight,
        "eval_dataset": eval_dataset,
        "n_classes": n_classes,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }


def remove_columns(ds, keep_columns=["label"]):
    return [column for column in ds.column_names if column not in keep_columns]


def preprocess_function(samples, tokenizer):
    return tokenizer(
        samples["text"],
        max_length=512,
        truncation=True,
    )


def load_and_preproc_data(project_name: str, tokenizer):

    data = load_data(project_name)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    eval_dataset = data["eval_dataset"]

    # Override preprocessing function
    def _preprocess_function(samples):
        return preprocess_function(samples, tokenizer)

    train_tokenized_ds = train_dataset.map(
        _preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=remove_columns(train_dataset),
    )

    val_tokenized_ds = val_dataset.map(
        _preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=remove_columns(val_dataset)
    )

    eval_tokenized_ds = eval_dataset.map(
        _preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=remove_columns(eval_dataset),
    )

    data.update({
        "eval_tokenized_ds": eval_tokenized_ds,
        "train_tokenized_ds": train_tokenized_ds,
        "val_tokenized_ds": val_tokenized_ds
    })

    return data


def set_logging():
    transformers.utils.logging.enable_progress_bar()
    transformers.utils.logging.set_verbosity_info()
    logger = transformers.utils.logging.get_logger("transformers")
    logger.info("INFO")


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding="max_length",
        add_prefix_space=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class WeightedCELossTrainer(Trainer):
    
    classes_weight = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        # loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.classes_weight, device=model.device, dtype=logits.dtype))
        # print(f"classes_weight: {self.classes_weight}")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.classes_weight.clone().detach().to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss