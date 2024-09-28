#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:34:00 2024

@author: vbeliveau
"""


import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # BitsAndBytesConfig,
)


if __name__ == "__main__":

    model_id = "RJuro/munin-neuralbeagle-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        trust_remote_code=False,
    )
