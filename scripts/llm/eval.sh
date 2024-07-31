#!/bin/bash

apptainer run \
	--net --network none \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/eval.py \
	--model $1 \
	--project $2 \
	"${@:3}"