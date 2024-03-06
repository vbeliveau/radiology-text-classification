#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:06:28 2024

@author: vbeliveau

Description:
    Script for preprocessing texts to train setfit classification models.
    Sentences with specific regular expresions will be extracted and combined in a single text.
"""

import re

import pandas as pd
import text_utils as tu

from pathlib import Path


nlp_model = tu.load_nlp_model()

# Load data
root_dir = "/proc_data1/bd5/nlp"
df_doccano = pd.read_csv(f"{root_dir}/data/anonymized_doccano_dataset.csv")
df_doccano["labels"] = [labels.split(",") for labels in df_doccano["labels"]]

# Define projects to be processed
projects = {
    # "FCD": "FCD - Helene",
    # "No_FCD_vs_rest": "FCD - Helene",
    # "MTS": "MTS - Helene",
    "hippocampus": "Hippocampus - Helene",
}


# Create output directory
out_dir = f"{root_dir}/data/preproc"
Path(out_dir).mkdir(parents=True, exist_ok=True)

# DEBUG
# project_name = "High_FCD_vs_No_FCD"

# %%

for project_name in projects:

    # %%

    # DEBUG
    print(f"Current project labels - {project_name}")
    print(set(sum([labels for labels in df_doccano.query(
        f"project == '{projects[project_name]}'")["labels"]], [])))

    if project_name == "FCD":
        regular_expression = "fcd|dyspla"
        labels = {  # consider added previous FCD
            "No FCD": ["No FCD"],
            "Possible FCD": ["Possible FCD"],
            "Highly probable FCD": ["Highly probable FCD"],
            "FCD": ["FCD"],
        }

    if project_name == "FCD_vs_rest":
        regular_expression = "fcd|dyspla"
        labels = {  # consider added previous FCD
            "No FCD": ["No FCD", "Possible FCD", "Highly probable FCD"],
            "FCD": ["FCD"],
        }

    if project_name == "No_FCD_vs_rest":
        regular_expression = "fcd|dyspla"
        labels = {  # consider added previous FCD
            "No FCD": ["No FCD"],
            "FCD": ["Possible FCD", "Highly probable FCD", "FCD"],
        }

    if project_name == "High_FCD_vs_No_FCD":
        regular_expression = "fcd|dyspla"
        labels = {  # consider added previous FCD
            "No FCD": ["No FCD"],
            "FCD": ["FCD", "Highly probable FCD"],
        }

    if project_name == "MTS":
        regular_expression = '(?<!a)mtl?s|(?<!Neurologisk Klinik )s(k|c)lero(?!(seklinik|se amb|SE RH))'
        labels = {
            "No MTS": ["No MTS"],
            "Potential MTS": ["Potential MTS", "Potential MTS Left", "Potential MTS Right"],
            "MTS": ["MTS", "MTS Left", "MTS Right"],
        }

    if project_name == "hippocampus":
        regular_expression = "hippoc|hipoc|hyppoc|hypoc"
        abnormal_labels = list(set(sum([labels for labels in df_doccano.query(
            f"project == '{projects[project_name]}'")["labels"]], [])))
        abnormal_labels.remove("Normal")
        abnormal_labels.remove("Normal brain")
        abnormal_labels.remove("To be checked")
        abnormal_labels.remove("Unspecified")
        abnormal_labels.remove("LITT")
        abnormal_labels.remove("Post operative")
        abnormal_labels.remove("Duplicate")

        # Question: there are "Normal brain" cases with neither "Normal" or "Unspecified" ... What would then be the best classification?

        labels = {
            "Normal": ["Normal"],
            "Abnormal": abnormal_labels,
        }

# %%

    # Identify any text with the matching labels

    df = None

    # DEBUG
    # label = "other"

    for label in labels:

        # DEBUG
        list(set(sum([labels for labels in df_doccano.query(
            f"project == '{projects[project_name]}'")["labels"]], [])))

        selected_labels = df_doccano[[
            any([label_ in text_labels for label_ in labels[label]])
            for text_labels in df_doccano["labels"]
        ]]["labels"]

        texts = df_doccano[[
            any([label_ in text_labels for label_ in labels[label]])
            for text_labels in df_doccano["labels"]
        ]]["text"]

        if len(texts) == 0:
            raise RuntimeError(f"No text matching label {label}")

        sentences, sentences_index = tu.extract_sentences(
            texts,
            nlp_model=nlp_model,
            batch_size=10,
            n_process=8,
            flatten=False,
            verbose=True,
        )
        sentences_list = tu.sentences_to_list(sentences, sentences_index)

        # Remove empty sentences
        for n in range(len(sentences_list)):
            sentences_list[n] = [sentence for sentence in sentences_list[n]
                                 if len(sentence.strip()) > 0]

        # Recursively merge sentences
        #   - with a single word with previous sentence
        #   - starting with (, : ;) with previous sentence
        #   - starting with a lower-case
        for sentences in sentences_list:
            merge_index = [n for n in range(len(sentences))
                           if len(sentences[n].split(" ")) <= 1 or
                           sentences[n].strip()[0] in [".", ",", ":", ";"] or
                           sentences[n].strip()[0].islower()]
            for n in merge_index[::-1]:
                sentences[(n-1):(n+1)] = [" ".join(sentences[(n-1):(n+1)])]

        # Filter out and concatenate sentences
        pattern = re.compile(
            regular_expression,
            flags=re.DOTALL | re.IGNORECASE | re.MULTILINE
        )

        filtered_texts = [
            " ".join([sentence for sentence in sentences
                      if pattern.search(sentence.lower()) is not None])
            for sentences in sentences_list
        ]

        # Aggregate texts across labels
        df = pd.concat([
            df,
            pd.DataFrame(
                {"label": label, "text": filtered_texts, "full_text": texts})
        ])

        # Save out data
        df.to_csv(f"{out_dir}/{project_name}.csv", index=False)
