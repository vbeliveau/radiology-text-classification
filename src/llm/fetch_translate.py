#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:34:00 2024

@author: vbeliveau
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer


if __name__ == "__main__":

    # Load Model
    model_name = 'google/madlad400-10b-mt'
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # tokenizer.add_tokens(AddedToken("\n", normalized=False))
