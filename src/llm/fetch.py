#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:34:00 2024

@author: vbeliveau
"""

import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # BitsAndBytesConfig,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    print(args.model)

    if args.model == "biomistral":
        model_id = "BioMistral/BioMistral-7B"

    if args.model == "llama3-70B":
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    if args.model == "munin":
        model_id = "RJuro/munin-neuralbeagle-7b"

    if args.token is not None:
        print(f"Using token {args.token}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=args.token,
        trust_remote_code=False,
    )
