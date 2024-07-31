#!/bin/bash
	
bash ${NLP_ROOT}/scripts/llm/eval.sh biomistral FCD \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10

bash ${NLP_ROOT}/scripts/llm/eval.sh biomistral MTS \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10

bash ${NLP_ROOT}/scripts/llm/eval.sh biomistral hippocampus \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10