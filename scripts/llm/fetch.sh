#!/bin/bash

apptainer run \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/fetch.py \
	--model $1 \
	"${@:2}"