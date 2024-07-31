#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:28:21 2023

@author: vbeliveau
"""


import argparse
import json

from setfit import SetFitModel, SetFitHead
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from utils import (
    load_data,
    plot_tsne,
    root_dir,
)


def evaluate_model(
        model_dir,
        project_name,
        data_dir=None,
        do_plot_tsne=False
):

    class CustomSetFitHead(SetFitHead):
        def __init__(self):
            super().__init__()

    data = load_data(project_name, data_dir=data_dir)
    category_names = data["category_names"]
    eval_dataset = data["eval_dataset"]
    y_true = eval_dataset["label"]

    model = SetFitModel.from_pretrained(model_dir)

    # Plot t-SNE, if required
    if do_plot_tsne:
        plot_tsne(model, eval_dataset, label_names=category_names)

    # Compute F1-score
    y_pred = model.predict(eval_dataset["text"]).cpu().numpy()
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1_score_macro = f1_score(y_true, y_pred, average="macro")
    metric = {
        "balanced_accuracy": balanced_accuracy,
        "f1_score_macro": f1_score_macro,
    }
    print(metric)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--eval_type", type=str, required=True)
    parser.add_argument("--tsne", action="store_true")
    args = parser.parse_args()

    # Load configs and assign parameters
    with open(args.config_json, "r") as f:
        configs = json.load(f)

    project_name = configs["project_name"]
    model_id = configs.get("model_id")
    model_str = configs.get("model_str", model_id.split("/")[-1])
    if args.eval_type == "end-to-end":
        model_dir = f"{root_dir}/models/setfit/{model_str}/{project_name}/end_to_end/best_model"
    elif args.eval_type == "head":
        model_dir = f"{root_dir}/models/setfit/{model_str}/{project_name}/head/best_model"
    data_dir = configs.get("data_dir", "/nlp/data/preproc")

    evaluate_model(
        model_dir,
        project_name,
        data_dir=data_dir,
        do_plot_tsne=args.tsne
    )
