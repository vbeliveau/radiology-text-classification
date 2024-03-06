#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:47:01 2024

@author: vbeliveau
"""


import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch

from datasets import Dataset
from matplotlib.lines import Line2D
from optuna import Trial
from pathlib import Path
from sentence_transformers import SentenceTransformer
from setfit import (
    SetFitHead,
    SetFitModel,
    Trainer,
    TrainingArguments,
)
from sklearn.manifold import TSNE
from tqdm import tqdm, trange
from torch import Tensor, nn
from torch.utils.data import DataLoader
from typing import List, Optional, Union


root_dir = "/nlp"
if not Path(root_dir).is_dir():
    root_dir = "/proc_data1/bd5/nlp"
    print(f"Changed root_dir from /nlp to {root_dir}")

default_batch_size = 128


class OptunaSetFitModelBodyTrainer(Trainer):

    # Trainer is modified to perform hyperparameter tuning with optuna for model body
    # https://github.com/huggingface/setfit/blob/1e3ce937e2e430c549b5b43ffb48b43d714efc24/src/setfit/trainer.py#L688

    def __init__(self, trial=None, **kwargs):
        super().__init__(**kwargs)
        self.trial = trial
        self.eval_loss = None

    def maybe_log_eval_save(
        self,
        model_body: SentenceTransformer,
        eval_dataloader: Optional[DataLoader],
        args: TrainingArguments,
        scheduler_obj,
        loss_func,
        loss_value: torch.Tensor,
    ) -> None:
        if self.control.should_log:
            learning_rate = scheduler_obj.get_last_lr()[0]
            metrics = {"embedding_loss": round(
                loss_value.item(), 4), "learning_rate": learning_rate}
            self.control = self.log(args, metrics)

        if self.control.should_evaluate and eval_dataloader is not None:
            self.eval_loss = self._evaluate_with_loss(
                model_body, eval_dataloader, args, loss_func)
            learning_rate = scheduler_obj.get_last_lr()[0]
            metrics = {"eval_embedding_loss": round(
                self.eval_loss, 4), "learning_rate": learning_rate}
            self.control = self.log(args, metrics)

            self.control = self.callback_handler.on_evaluate(
                args, self.state, self.control, metrics)

            loss_func.zero_grad()
            loss_func.train()

            if self.trial is not None:
                self.trial.report(
                    self.eval_loss, self.state.global_step * self.args.batch_size[0])
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()


class OptunaSetFitEndToEndTrainer(Trainer):

    # Trainer is modified to perform hyperparameter tuning with optuna for end-to-end model
    # https://github.com/huggingface/setfit/blob/1e3ce937e2e430c549b5b43ffb48b43d714efc24/src/setfit/trainer.py#L773

    def __init__(self, trial=None, **kwargs):
        super().__init__(**kwargs)
        self.trial = trial

    def train_classifier(
        self, x_train: List[str], y_train: Union[List[int], List[List[int]]], args: Optional[TrainingArguments] = None
    ) -> None:
        """
        Method to perform the classifier phase: fitting a classifier head.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
        """
        args = args or self.args or TrainingArguments()

        self.model.fit(
            self,
            self.trial,
            x_train,
            y_train,
            num_epochs=args.classifier_num_epochs,
            batch_size=args.classifier_batch_size,
            body_learning_rate=args.body_classifier_learning_rate,
            head_learning_rate=args.head_learning_rate,
            l2_weight=args.l2_weight,
            max_length=args.max_length,
            show_progress_bar=args.show_progress_bar,
            end_to_end=args.end_to_end,
        )


class WeightedCELossSetFitHead(SetFitHead):

    # SetFitHead is modified to enable weighted CE loss
    # https://github.com/huggingface/setfit/blob/1e3ce937e2e430c549b5b43ffb48b43d714efc24/src/setfit/modeling.py#L165

    def __init__(self, weight=None, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)
        self.weight = weight.to(self.device)

    def get_loss_fn(self) -> nn.Module:
        print(
            f"Using the following weights for the loss function: {self.weight}")
        if self.multitarget:  # if sigmoid output
            return torch.nn.BCEWithLogitsLoss(weight=self.weight)
        return torch.nn.CrossEntropyLoss(weight=self.weight)


class OptunaSetFitEndToEndModel(SetFitModel):

    # SetFitModel is modified to enable online monitoring of end-to-end training when
    # optimizing with optuna
    # https://github.com/huggingface/setfit/blob/1e3ce937e2e430c549b5b43ffb48b43d714efc24/src/setfit/modeling.py#L200

    def fit(
        self,
        trainer: Trainer,
        trial: Trial,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        num_epochs: int,
        batch_size: Optional[int] = None,
        body_learning_rate: Optional[float] = None,
        head_learning_rate: Optional[float] = None,
        end_to_end: bool = False,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """Train the classifier head, only used if a differentiable PyTorch head is used.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            num_epochs (`int`): The number of epochs to train for.
            batch_size (`int`, *optional*): The batch size to use.
            body_learning_rate (`float`, *optional*): The learning rate for the `SentenceTransformer` body
                in the `AdamW` optimizer. Disregarded if `end_to_end=False`.
            head_learning_rate (`float`, *optional*): The learning rate for the differentiable torch head
                in the `AdamW` optimizer.
            end_to_end (`bool`, defaults to `False`): If True, train the entire model end-to-end.
                Otherwise, freeze the `SentenceTransformer` body and only train the head.
            l2_weight (`float`, *optional*): The l2 weight for both the model body and head
                in the `AdamW` optimizer.
            max_length (`int`, *optional*): The maximum token length a tokenizer can generate. If not provided,
                the maximum length for the `SentenceTransformer` body is used.
            show_progress_bar (`bool`, defaults to `True`): Whether to display a progress bar for the training
                epochs and iterations.
        """
        if self.has_differentiable_head:  # train with pyTorch
            self.model_body.train()
            self.model_head.train()
            if not end_to_end:
                self.freeze("body")

            dataloader = self._prepare_dataloader(
                x_train, y_train, batch_size, max_length)
            criterion = self.model_head.get_loss_fn()
            optimizer = self._prepare_optimizer(
                head_learning_rate, body_learning_rate, l2_weight)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5)

            trainer.state.global_step = 0
            trainer.control = trainer.callback_handler.on_train_begin(
                trainer.args, trainer.state, trainer.control)
            steps_per_epoch = len(dataloader)

            for epoch_idx in trange(num_epochs, desc="Epoch", disable=not show_progress_bar):

                trainer.control = trainer.callback_handler.on_epoch_begin(
                    trainer.args, trainer.state, trainer.control)

                batch_loss = []

                for n_step, batch in tqdm(enumerate(dataloader), desc="Iteration", disable=not show_progress_bar, leave=False):

                    trainer.control = trainer.callback_handler.on_step_begin(
                        trainer.args, trainer.state, trainer.control)

                    features, labels = batch
                    optimizer.zero_grad()

                    # to model's device
                    features = {k: v.to(self.device)
                                for k, v in features.items()}
                    labels = labels.to(self.device)

                    outputs = self.model_body(features)
                    if self.normalize_embeddings:
                        outputs["sentence_embedding"] = nn.functional.normalize(
                            outputs["sentence_embedding"], p=2, dim=1
                        )
                    outputs = self.model_head(outputs)
                    logits = outputs["logits"]

                    loss: torch.Tensor = criterion(logits, labels)

                    # Log loss
                    batch_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    trainer.state.global_step += 1
                    trainer.state.epoch = epoch_idx + \
                        (n_step + 1) / steps_per_epoch

                    trainer.control = trainer.callback_handler.on_step_end(
                        trainer.args, trainer.state, trainer.control)
                    trainer.control = trainer.log(
                        trainer.args, {"batch_loss": batch_loss[-1]})

                # Keep track of best metric
                metric = trainer.evaluate()["metric"]
                trainer.control = trainer.log(
                    trainer.args, {"eval_metric": metric})
                # if self.best_metric is None or metric > self.best_metric:
                #     self.best_metric = metric

                self.model_body.train()
                self.model_head.train()

                trial.report(metric, epoch_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                scheduler.step()

                trainer.control = trainer.callback_handler.on_epoch_end(
                    trainer.args, trainer.state, trainer.control)
                trainer.control = trainer.log(
                    trainer.args, {"epoch_loss": np.mean(batch_loss)})

            if not end_to_end:
                self.unfreeze("body")
        else:  # train with sklearn
            embeddings = self.model_body.encode(
                x_train, normalize_embeddings=self.normalize_embeddings)
            self.model_head.fit(embeddings, y_train)
            if self.labels is None and self.multi_target_strategy is None:
                # Try to set the labels based on the head classes, if they exist
                # This can fail in various ways, so we catch all exceptions
                try:
                    classes = self.model_head.classes_
                    if classes.dtype.char == "U":
                        self.labels = classes.tolist()
                except Exception:
                    pass


def load_data(project_name):

    df = pd.read_csv(
        f"{root_dir}/data/preproc/{project_name}.csv")
    # Remove nans.. should not be happening at this stage
    df = df[[isinstance(text, str) for text in df["text"]]]

    # Convert labels to numerical categories
    factors = pd.factorize(df['label'])
    category_names = list(factors[1])
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
        "category_names": category_names,
        "classes_count": classes_count,
        "classes_weight": classes_weight,
        "eval_dataset": eval_dataset,
        "n_classes": n_classes,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }


def plot_tsne(model, data, label_names=None):

    # Get embeddings
    embeddings = model.encode(data["text"], batch_size=32)
    if embeddings.device != "cpu":
        embeddings = embeddings.cpu().numpy()

    # Apply t-SNE
    X_embedded = TSNE(
        n_components=2,
        random_state=42,
        init="random",
        perplexity=100,
        # n_iter=300,
    ).fit_transform(embeddings)

    # Organize embeddings in a DataFrame
    df_embeddings = pd.DataFrame(X_embedded, columns=["x", "y"])
    df_embeddings["label"] = data["label"]

    # Visualize the embeddings
    plt.figure(figsize=(15, 10))

    # Compute unique classes in test set
    unique_classes = list(set(data["label"]))

    # Generate a unique color for each possible label
    colors = plt.cm.Set1(range(len(unique_classes)))

    # Define the 5 marker styles
    markers = ['o', 's', '^', 'D', 'P']

    # Plot each point
    for row in df_embeddings.itertuples():
        point_color = colors[row.label]
        point_marker = markers[row.label]
        plt.scatter(
            row.x, row.y,
            color=point_color,
            marker=point_marker,
            s=100
        )

    # Creating custom legend
    if label_names is not None:
        legend_elements = [
            Line2D([0], [0],
                   marker=markers[i],
                   color='w',
                   label=label_names[i],
                   markersize=10,
                   markerfacecolor=colors[i])
            for i in range(len(unique_classes))
        ]

        plt.legend(handles=legend_elements,
                   loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('t-SNE visualization of sentence embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()