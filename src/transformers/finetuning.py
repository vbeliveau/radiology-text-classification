#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:30:35 2023

@author: vbeliveau
"""


import argparse
import json
import os
import datasets
import transformers

from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from utils import (
    load_tokenizer,
    preprocess_function,
    set_logging,
)


def main():

    parser = argparse.ArgumentParser(prog="Pre-Training of Transformers using Huggingface")
    parser.add_argument('-c', '--config_json', type=str, default=None,
            help="Configuration JSON file specifying options for fine-tuning."
        )
    args = parser.parse_args()
    with open(args.config_json, "r") as f:
        configs = json.load(f)

    datasets = configs["datasets"].split(",")
    model_id = configs["model_name"]
    training_parser = HfArgumentParser(TrainingArguments)
    training_args, = training_parser.parse_json_file(json_file=configs["training_configs"])

    print(f"Model: {model_id}")
    print(f"Datasets: {datasets}")

    set_logging()

    # Define paths
    root_dir = "/nlp"
    preproc_dir = f"{root_dir}/preproc"

    tokenizer = load_tokenizer(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )

    # %% Load and preprocess data

    print("Loading data")

    data_files =[]

    if "clinical_notes" in datasets:
        data_files += [f"/nlp/data/anonymized_clinical_notes_part{part}.csv" for part in [1, 2, 3]]

    # if "eeg_reports" in datasets:
    #     data_files.append("/nlp/data/anonymized_radiology_texts.json")

    if "radiology_reports" in datasets:
        data_files += [f"/nlp/data/anonymized_radiology_reports_part{part}.csv" for part in [1, 2, 3]]

    if len(data_files) == 0:
        raise RuntimeError("No valid dataset specified.")

    print(data_files)
    dataset = load_dataset("csv", data_files=data_files, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Override preprocessing function
    def _preprocess_function(samples):
        return preprocess_function(samples, tokenizer)

    tokenized_ds = dataset.map(
        _preprocess_function,
        batched=True,
        num_proc=32,
        remove_columns=dataset["train"].column_names,
    )

    PREFIX_CHECKPOINT_DIR = "checkpoint"
    TRAINER_STATE_NAME = "trainer_state.json"

    # Fix for the following issue: https://github.com/huggingface/transformers/issues/27925
    class CustomTrainer(Trainer):

        def _save_checkpoint(self, model, trial, metrics=None):
            # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
            # want to save except FullyShardedDDP.
            # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
                logger.warning(
                    f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                    "Saving will proceed but saved results may be invalid."
                )
                staging_output_dir = output_dir
            else:
                staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
            self.save_model(staging_output_dir, _internal_call=True)

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(staging_output_dir)
                # Save RNG state
                self._save_rng_state(staging_output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(staging_output_dir)

            # Place checkpoint in final location after all saving is finished.
            # First wait for everyone to finish writing
            self.args.distributed_state.wait_for_everyone()
            # Then go through the rewriting process starting on process 0
            if staging_output_dir != output_dir:

                if self.args.distributed_state.is_local_main_process if self.args.save_on_each_node else self.args.distributed_state.is_main_process:

                    print("Renaming model checkpoint folder to true location")

                    if os.path.exists(staging_output_dir):
                        os.rename(staging_output_dir, output_dir)

            self.args.distributed_state.wait_for_everyone()

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


    trainer = CustomTrainer(
        args=training_args,
        eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        model=model,        
        train_dataset=tokenized_ds["train"],       
    )

    # Resume if any checkpoint exists, otherwise start training
    if len(list(Path(training_args.output_dir).glob("checkpoint*"))) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(training_args["output_dir"])


if __name__ == "__main__":
    main()