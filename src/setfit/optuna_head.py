#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:28:21 2023

@author: vbeliveau
"""

import argparse
import json
import os
import optuna

from optuna.storages import RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from pathlib import Path
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from utils import (
    load_data,
    OptunaSetFitEndToEndModel,
    OptunaSetFitEndToEndTrainer,
    root_dir,
    WeightedCELossSetFitHead,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Optuna hyperparameters tuning for Setfit model body.')
    # parser.add_argument("-c", "--config_json", type=str, required=True)
    parser.add_argument("-c", "--config_json", type=str, required=True)
    args = parser.parse_args()

    # Load configs and assign parameters
    with open(args.config_json, "r") as f:
        configs = json.load(f)

    project_name = configs.get("project_name", None)
    if project_name is None:
        raise ValueError("project_name is not defined in config file.")
    model_id = configs.get("model_id")
    model_str = configs.get("model_str", model_id.split("/")[-1])

    os.chdir(root_dir)
    output_dir = f"{root_dir}/models/setfit/{model_str}/{project_name}/head/optuna"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = configs.get("data_dir", "/nlp/data/preproc")
    data = load_data(project_name, data_dir=data_dir)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    n_classes = data["n_classes"]
    classes_weight = data["classes_weight"]

    num_epochs = 50

    class MaximumNumberOfTrialsException(Exception):
        pass

    def objective(trial):

        n_complete = len([t for t in study.trials
                          if t.state == TrialState.COMPLETE or
                          t.state == TrialState.RUNNING or
                          t.state == TrialState.PRUNED])
        print(f"Trials completed: {n_complete}")

        if n_complete > n_trials:  # current trial counts as running
            study.stop()
            raise MaximumNumberOfTrialsException

        # Head learning rate
        head_learning_rate_min = configs.get("head_learning_rate_min")
        head_learning_rate_max = configs.get("head_learning_rate_max")
        head_learning_rate_log = configs.get("head_learning_rate_log")

        if head_learning_rate_min is not None and head_learning_rate_max is not None:
            head_learning_rate = trial.suggest_float(
                "head_learning_rate",
                head_learning_rate_min,
                head_learning_rate_max,
                log=head_learning_rate_log)
        else:
            head_learning_rate = 1e-6

        # Batch size
        batch_size = configs.get("batch_size", 128)
        if batch_size != 128:
            batch_size_str = f"_batch-size-{batch_size}"
        else:
            batch_size_str = ""

        # num_epochs
        num_epochs = configs.get("num_epochs", 50)
        print(num_epochs)
        if isinstance(num_epochs, str):
            n_epochs_list = [int(num_epoch)
                             for num_epoch in num_epochs.split(",")]
            num_epochs = trial.suggest_categorical("num_epochs", n_epochs_list)
            num_epochs_str = f"_num-epochs-{num_epochs}"
        else:
            num_epochs_str = ""

        model_prefix = f"trial-{trial.number}_head-learning-rate-{head_learning_rate:.2e}{batch_size_str}{num_epochs_str}"
        print(model_prefix)

        training_args = TrainingArguments(
            batch_size=batch_size,
            end_to_end=False,
            head_learning_rate=head_learning_rate,
            logging_dir=f"{output_dir}/runs/{model_prefix}",
            logging_steps=1,
            logging_strategy="steps",
            max_length=128,
            num_epochs=num_epochs,
            report_to=["tensorboard"],
            sampling_strategy="oversampling",
            use_amp=True,
        )

        model_body = SentenceTransformer(model_id)

        model_head = WeightedCELossSetFitHead(
            in_features=model_body.get_sentence_embedding_dimension(),
            out_features=n_classes,
            weight=classes_weight,
        )

        model_body.to(model_head.device)

        model = OptunaSetFitEndToEndModel(model_body, model_head)

        # Assign metric
        metric_type = configs.get("metric", "f1_macro")
        print(f"metric_type: {metric_type}")
        # metric = None
        if metric_type == "f1_macro":
            def metric(x, y): return f1_score(x, y, average="macro")
        try:
            metric
        except:
            raise ValueError(
                "No valid metric specified in configuration file.")

        trainer = OptunaSetFitEndToEndTrainer(
            args=training_args,
            metric=metric,
            eval_dataset=val_dataset,
            # model_init=model_init,
            model=model,
            trial=trial,
        )

        trainer.train_classifier(
            x_train=train_dataset["text"],
            y_train=train_dataset["label"],
        )

        # The "best_metric" might peak high but not be very stable and will
        # therefore not generalize very well. Instead we return the final
        # evaluation of the converged model.
        # return model.best_metric
        return trainer.evaluate()["metric"]

    # Setup storage
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    db_path = f"{output_dir}/db.sqlite3"
    storage_path = f"sqlite:///{db_path}"
    print(storage_path)

    storage = optuna.storages.RDBStorage(
        failed_trial_callback=RetryFailedTrialCallback(max_retry=None),
        grace_period=120,
        heartbeat_interval=60,
        url=storage_path,
    )

    # Create study
    study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
        storage=storage,
        study_name=project_name,
    )

    # Optimize
    n_trials = 100
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[MaxTrialsCallback(n_trials, states=(
            TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING))],
        catch=(MaximumNumberOfTrialsException,),
    )

    best_params = study.best_params
    best_params["project_name"] = project_name
    if "num_epochs" not in list(best_params.keys()):
        best_params["num_epochs"] = num_epochs
    if "batch_size" not in list(best_params.keys()):
        best_params["batch_size"] = configs.get("batch_size", 128)

    # Print out best parameters
    print(best_params)

    # Save out parameters
    json_file = f"{output_dir}/best_params.json"
    with open(json_file, "w") as f:
        json.dump(best_params, f)
