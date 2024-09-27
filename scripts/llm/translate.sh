#!/bin/bash

# Example usage:
# bash translate.sh /nlp/data/preproc-melba/hippocampus.csv /nlp/data/preproc-melba-translated/hippocampus.csv

apptainer run \
	--net --network none \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/translate.py \
	--in_csv $1 \
	--out_csv $2