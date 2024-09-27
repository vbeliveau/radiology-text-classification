#!/bin/bash

apptainer run \
	--net --network none \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/neuralbeagle_eval.py \
	"$@"