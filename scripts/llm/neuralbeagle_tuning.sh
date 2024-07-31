#!/bin/bash
NLP_ROOT="/proc_data1/bd5/nlp"
apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/neuralbeagle_tuning.py -project $1 \
    "${@:2}"