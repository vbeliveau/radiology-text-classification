#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:28:21 2023

@author: vbeliveau
"""


import argparse
import json
import optuna
import os

from optuna.storages import RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer_utils import HPSearchBackend
from typing import Dict

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

    # Assign paths
    os.chdir(root_dir)
    output_dir = training_args_dict["output_dir"] + "/optuna"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    set_logging()

    # Load tokenizer and define data collator
    tokenizer = load_tokenizer(base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load and preprocess data
    data = load_and_preproc_data(project_name, tokenizer)
    train_tokenized_ds = data["train_tokenized_ds"]
    val_tokenized_ds = data["val_tokenized_ds"]
    n_classes = data["n_classes"]
    classes_weight = data["classes_weight"]

    def compute_objective(metrics: Dict[str, float]) -> float:
        return metrics["eval_f1_score_macro"]

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

        # Assign optuna paremeters
        learning_rate_min = configs.get("learning_rate_min")
        learning_rate_max = configs.get("learning_rate_max")
        learning_rate_log = configs.get("learning_rate_log")
        learning_rate = trial.suggest_float(
            "learning_rate",
            learning_rate_min,
            learning_rate_max,
            log=learning_rate_log)

        weight_decay_min = configs.get("weight_decay_min")
        weight_decay_max = configs.get("weight_decay_max")
        weight_decay_log = configs.get("weight_decay_log")
        weight_decay = trial.suggest_float(
            "weight_decay",
            weight_decay_min,
            weight_decay_max,
            log=weight_decay_log)

        training_args_dict["logging_dir"] = f"{output_dir}/runs/trial-{trial.number}_learning-rate-{learning_rate:.2e}_weight-decay-{weight_decay:.2e}"
        training_parser = HfArgumentParser(TrainingArguments)
        training_args, = training_parser.parse_dict(training_args_dict)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=n_classes)

        if configs.get("peft", False):
            print("Running using PEFT")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=configs.get("peft_r", 2),
                lora_alpha=configs.get("peft_lora_alpha", 16),
                lora_dropout=configs.get("peft_lora_dropout", 0.1),
                bias=configs.get("peft_bias", "none"),
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        trainer = WeightedCELossTrainer(
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            eval_dataset=val_tokenized_ds,
            model=model,
            train_dataset=train_tokenized_ds,
        )
        trainer.classes_weight = classes_weight

        def optuna_hp_space(trial):
            return {"learning_rate": learning_rate,
                    "weight_decay": weight_decay}

        trainer.hp_search_backend = HPSearchBackend.OPTUNA
        trainer.hp_space = optuna_hp_space
        trainer.compute_objective = compute_objective

        trainer.train(trial=trial)

        return trainer.evaluate()["eval_f1_score_macro"]

    # Setup storage
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    storage_path = f"sqlite:///{output_dir}/db.sqlite3"
    print(storage_path)

    storage = optuna.storages.RDBStorage(
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
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
    n_complete = len([t for t in study.trials
                      if t.state == TrialState.COMPLETE or
                      t.state == TrialState.RUNNING or
                      t.state == TrialState.PRUNED])
    print(f"Trials completed: {n_complete}")

    if n_complete < n_trials:  # current trial counts as running
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[MaxTrialsCallback(n_trials, states=(
                TrialState.COMPLETE, TrialState.RUNNING))],
            catch=(MaximumNumberOfTrialsException,),
        )
    best_params = study.best_params
    best_params["project_name"] = project_name
    print(best_params)

    # Save out parameters
    json_file = f"{output_dir}/best_params.json"
    with open(json_file, "w") as f:
        json.dump(best_params, f)
