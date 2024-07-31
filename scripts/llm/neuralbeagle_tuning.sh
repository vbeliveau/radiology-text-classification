#!/bin/bash

apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/neuralbeagle_tuning.py -project $1 \
    "${@:2}"