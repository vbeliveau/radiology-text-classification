#!/bin/bash

bash ${NLP_ROOT}/scripts/llm/neuralbeagle_eval.sh --project FCD --data_dir /nlp/data/preproc-melba
bash ${NLP_ROOT}/scripts/llm/neuralbeagle_eval.sh --project MTS --data_dir /nlp/data/preproc-melba
bash ${NLP_ROOT}/scripts/llm/neuralbeagle_eval.sh --project HA --data_dir /nlp/data/preproc-melba