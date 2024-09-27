#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:28:21 2023

@author: vbeliveau
"""

import argparse
import os
import json
import optuna

# from datetime import datetime
from optuna.storages import RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from pathlib import Path
from setfit import (
    # sample_dataset,
    SetFitModel,
    TrainingArguments,
)
from utils import (
    load_data,
    OptunaSetFitModelBodyTrainer,
    root_dir
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Optuna hyperparameters tuning for Setfit model body.')
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
    output_dir = f"{root_dir}/models/setfit/{model_str}/{project_name}/model_body/optuna"
    print(f"Output directory: {output_dir}")

    # Load data
    data_dir = configs.get("data_dir", "/nlp/data/preproc")
    data = load_data(project_name, data_dir=data_dir)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]

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

        body_learning_rate_min = configs.get("body_learning_rate_min")
        body_learning_rate_max = configs.get("body_learning_rate_max")
        body_learning_rate_log = configs.get("body_learning_rate_log")

        body_learning_rate = trial.suggest_float(
            "body_learning_rate",
            body_learning_rate_min,
            body_learning_rate_max,
            log=body_learning_rate_log)

        batch_size_min = configs.get("batch_size_min", None)
        batch_size_max = configs.get("batch_size_max", None)
        if batch_size_min is not None and batch_size_max is not None:
            batch_size = trial.suggest_int(
                "batch_size",
                batch_size_min,
                batch_size_max)
        else:
            batch_size = configs.get("batch_size", 8)

        model_prefix = f"trial-{trial.number}_body-learning-rate-{body_learning_rate:.2e}_batch-size-{batch_size}"
        print(model_prefix)

        total_train_examples = 25000
        total_eval_steps = 10

        training_args = TrainingArguments(
            batch_size=batch_size,
            body_learning_rate=body_learning_rate,
            evaluation_strategy="steps",
            eval_steps=total_train_examples/total_eval_steps//batch_size,
            logging_dir=f"{output_dir}/runs/{model_prefix}",
            logging_steps=1,
            max_length=128,
            max_steps=total_train_examples//batch_size,
            # num_epochs=1,  # overridden by max_steps
            report_to=["tensorboard"],
            sampling_strategy="oversampling",
            use_amp=True,
        )

        # There is some strange behavior with model_init, so using this
        # When using model_init, the model seems not to get properly reset
        # before new training.

        model = SetFitModel.from_pretrained(model_id)

        # There is a bug when using containers. Somehow config gets added
        # as an attribute, but is dict instead of PretrainedConfig.
        # Trainer tries to use the to_json_string() mehtod on dict, which
        # throws and error.
        if hasattr(model, "config"):
            delattr(model, "config")

        trainer = OptunaSetFitModelBodyTrainer(
            args=training_args,
            metric="accuracy",
            model=model,
            trial=trial,
        )

        trainer.train_embeddings(
            x_train=train_dataset["text"],
            y_train=train_dataset["label"],
            x_eval=val_dataset["text"],
            y_eval=val_dataset["label"],
        )

        # The "best_metric" might peak but not be very stable will
        # therefore to not generalize very well. Instead we return the final
        # evaluation of the converged model.
        return trainer.eval_loss

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
        direction="minimize",
        load_if_exists=True,
        storage=storage,
        study_name=project_name,
    )

    # Optimize
    n_trials = configs.get("n_trials", 100)
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[MaxTrialsCallback(n_trials, states=(
            TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING))],
        catch=(MaximumNumberOfTrialsException,),
    )

    best_params = study.best_params
    best_params["project_name"] = project_name
    if "batch_size" not in list(best_params.keys()):
        best_params["batch_size"] = configs.get("batch_size", 128)

    # Print out best parameters
    print(best_params)

    # Save out parameters
    json_file = f"{output_dir}/best_params.json"
    with open(json_file, "w") as f:
        json.dump(best_params, f)
