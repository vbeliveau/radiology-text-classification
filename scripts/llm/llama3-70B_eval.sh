#!/bin/bash
	
bash ${NLP_ROOT}/scripts/llm/eval.sh llama3-70B FCD \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10

bash ${NLP_ROOT}/scripts/llm/eval.sh llama3-70B MTS \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10

bash ${NLP_ROOT}/scripts/llm/eval.sh llama3-70B hippocampus \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10