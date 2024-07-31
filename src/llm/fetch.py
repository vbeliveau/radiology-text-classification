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

# Meditron llama3


class DebugArgs():
    def __init__(self):
        self.model = "llama3"
        self.token = "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"


args = DebugArgs()

# %%


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

# %%

    print(args.model)

    if args.model == "biomistral":
        model_id = "BioMistral/BioMistral-7B"

    if args.model == "llama3-8B":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    if args.model == "llama3-70B":
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    if args.model == "meditron-70B":
        model_id = "epfl-llm/meditron-70b"

    if args.model == "munin":
        model_id = "RJuro/munin-neuralbeagle-7b"

    if args.token is not None:
        print(f"Using token {args.token}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.token)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # load_in_4bit=True,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=args.token,
        trust_remote_code=False,
    )
