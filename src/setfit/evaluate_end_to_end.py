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
    ConfusionMatrixDisplay,
    f1_score,
)
from utils import (
    load_data,
    plot_tsne,
    root_dir,
)


# project_name = "MTS"


def evaluate_end_to_end(project_name, do_plot_tsne=False):

    class CustomSetFitHead(SetFitHead):
        def __init__(self):
            super().__init__()

    data = load_data(project_name)
    category_names = data["category_names"]
    eval_dataset = data["eval_dataset"]
    y_true = eval_dataset["label"]

    model = SetFitModel.from_pretrained(
        f"{root_dir}/models/setfit/{project_name}/end_to_end/best_model",
    )

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

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()

    # plt.savefig(
    #     f"{root_dir}/models/setfit/{project_name}/end_to_end/confusion_matrix.png", dpi=600)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Fine-tuining of SetFit model body.')
    parser.add_argument("-project_name", "--project_name", type=str, required=True)
    parser.add_argument("-tsne", "--tsne", action="store_true")
    args = parser.parse_args()

    # Load configs and assign parameters
    # with open(args.config_json, "r") as f:
    #     configs = json.load(f)

    # project_name = configs["project_name"]
    evaluate_end_to_end(args.project_name, do_plot_tsne=args.tsne)
