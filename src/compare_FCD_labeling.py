#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:28:21 2023

@author: vbeliveau

Description:
    Script for extracting labels from doccano projects. Full anonymized texts will be added.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import numpy as np
import pandas as pd
import shutil
import sqlite3


from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
)

# %%


def extract_doccano_data(doccano_dir, project_name):

    # Connect to doccano database
    src_db = f"{doccano_dir}/db.sqlite3"
    tmp_db = f"{doccano_dir}/db.sqlite3.tmp"
    shutil.copy(src_db, tmp_db)
    con = sqlite3.connect(tmp_db)
    con.row_factory = lambda cursor, row: row[0]
    cur = con.cursor()

    print(f"Processing: {project_name}")

    # Get project ID
    project_id = cur.execute(
        f''' SELECT id FROM projects_project WHERE name="{project_name}"; ''').fetchone()
    if project_id is None:
        raise ValueError(f'Project {project_name} could not be found.')

    # Extract project labels
    labels = cur.execute(
        f''' SELECT text FROM label_types_categorytype WHERE project_id={project_id}; ''').fetchall()

    df_labels = None

    # DEBUG
    # label = "MTS"

    for label in labels:

        # Get the label id
        label = label.replace('"', "'")  # replace quotes, if any
        label_id = cur.execute(f''' 
            SELECT id FROM label_types_categorytype 
            WHERE project_id={project_id} AND text="{label}";
        ''').fetchone()

        # Extract reports IDs for the current label
        # cur.execute(f''' SELECT * FROM labels_category; ''').description
        examples_id = cur.execute(f''' 
            SELECT example_id FROM labels_category
            WHERE label_id={label_id};
        ''').fetchall()

        # Get PDF files
        # cur.execute(f''' SELECT * FROM examples_example; ''').description
        examples_id_text = ', '.join([str(example_id)
                                      for example_id in examples_id])
        metas = cur.execute(f''' 
            SELECT meta FROM examples_example
            WHERE id IN ({examples_id_text});
        ''').fetchall()
        pdf_files = [eval(meta)['pdf_file'] for meta in metas]

        # Merge information
        df_ = pd.DataFrame({'label': label, "pdf_file": pdf_files})
        df_labels = pd.concat([df_labels, df_], axis=0)

    # %

    df_concat = None
    pdf_files = df_labels["pdf_file"].unique()

    for pdf_file in pdf_files:
        labels = ",".join(df_labels.query(
            "pdf_file == @pdf_file")["label"].unique())
        df_ = pd.DataFrame({
            "project": [project_name],
            "labels": [labels],
            "pdf_file": [pdf_file],

        }, index=[0])
        df_concat = pd.concat([df_concat, df_])

    con.close()

    return df_concat

# %%


def main():

    print("Create project dataset from doccano labels.")

    parser = argparse.ArgumentParser(prog='Process text from classification.')
    parser.add_argument("-doccano_dir", "--doccano_dir",
                        type=str, required=True)
    args = parser.parse_args()

    print("Extracting data")
    # doccano_dir = "/proc_data1/bd5/doccano"
    # df = extract_doccano_data(args.doccano_dir, "FCD - Helene")
    df_helene = extract_doccano_data(args.doccano_dir, "FCD - Helene")
    df_helene = df_helene.rename(columns={"labels": "labels_helene"})
    df_martin = extract_doccano_data(args.doccano_dir, "FCD - Martin")
    df_martin = df_martin.rename(columns={"labels": "labels_martin"})

    # Merge df
    df = df_helene.merge(df_martin, on="pdf_file")[
        ["labels_helene", "labels_martin"]]

    # Filter out labels

    selected_labels = ["No FCD", "Possible FCD", "Highly probable FCD", "FCD"]

    def filter_labels(labels_list):

        labels = [[label for label in labels.split(",") if labels in selected_labels]
                  for labels in labels_list]
        for label in labels:
            if len(label) > 1:
                raise ValueError("Too many labels detected.")
        labels = [label[0] if len(label) == 1 else "Not relevant"
                  for label in labels]
        return labels

    df["labels_helene"] = filter_labels(df["labels_helene"])
    df["labels_martin"] = filter_labels(df["labels_martin"])
    df["labels_helene"].unique()
    df["labels_martin"].unique()

    # Plot confusion matrix
    cm = confusion_matrix(df["labels_helene"],
                          df["labels_martin"], labels=selected_labels)
    print(cm)

    cohen_k = cohen_kappa_score(
        df["labels_helene"], df["labels_martin"], labels=selected_labels)
    print(f"Cohen's kappa: {cohen_k:.2}")

    perc_agreement = np.sum(np.diag(cm))/np.sum(cm[:])*100
    print(f"Percentage agreement: {perc_agreement:.1f}%")


if __name__ == '__main__':
    main()
