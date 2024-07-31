#!/bin/bash
NLP_ROOT="/proc_data1/bd5/nlp"
apptainer run \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/fetch.py \
	--model $1 \
	"${@:2}"