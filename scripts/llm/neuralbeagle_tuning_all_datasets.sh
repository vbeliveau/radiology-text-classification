#!/bin/bash

bash ${NLP_ROOT}/scripts/llm/neuralbeagle_tuning.sh FCD --min_samples 1 --max_samples 7 --data_dir /nlp/data/preproc-melba
bash ${NLP_ROOT}/scripts/llm/neuralbeagle_tuning.sh MTS --min_samples 1 --max_samples 7 --data_dir /nlp/data/preproc-melba
bash ${NLP_ROOT}/scripts/llm/neuralbeagle_tuning.sh HA --min_samples 1 --max_samples 7 --data_dir /nlp/data/preproc-melba