#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:34:00 2024

@author: vbeliveau
"""

import argparse
import pandas as pd

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


# class DebugArgs():

#     def __init__(self):
#         self.data_dir = "/proc_data1/bd5/nlp/data/preproc-midl"
#         self.project = "FCD"


# args = DebugArgs()


# %%


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

# %%

    # Read data
    df = pd.read_csv(args.in_csv)

    # Load Model
    model_name = 'google/madlad400-10b-mt'
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Translate texts
    print("Translating texts")
    translated_texts = []
    for danish_text in tqdm(df["text"], total=len(df)):
        text = f"<2en> {danish_text}"
        input_ids = tokenizer(
            text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=1024)
        translated_texts += [tokenizer.decode(outputs[0],
                                              skip_special_tokens=True)]

    # Save out
    df["text"] = translated_texts
    df.to_csv(args.out_csv, index=False)
