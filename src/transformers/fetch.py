#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:30:35 2023

@author: vbeliveau
"""


import argparse
import json

from transformers import AutoModelForMaskedLM
from utils import load_tokenizer


def main():

    parser = argparse.ArgumentParser(
        prog="Pre-Training of Transformers using Huggingface")
    parser.add_argument('-c', '--config_json', type=str, default=None,
                        help="Configuration JSON file specifying options for fine-tuning."
                        )
    args = parser.parse_args()
    with open(args.config_json, "r") as f:
        configs = json.load(f)

    model_id = configs["base_model"]
    load_tokenizer(model_id)
    AutoModelForMaskedLM.from_pretrained(model_id)


if __name__ == "__main__":
    main()
