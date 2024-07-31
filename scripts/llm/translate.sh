#!/bin/bash

apptainer run \
	--net --network none \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/translate.py \
	--in_csv $1 \
	--out_csv $2