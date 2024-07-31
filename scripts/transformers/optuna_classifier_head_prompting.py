#!/bin/bash
apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	transformers.sif \
	/nlp/src/transformers/optuna_classifier_head_prompting.py -c $1