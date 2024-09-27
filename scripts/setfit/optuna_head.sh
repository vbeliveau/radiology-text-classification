#!/bin/bash

apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	setfit.sif \
	/nlp/src/setfit/optuna_head.py -c $1