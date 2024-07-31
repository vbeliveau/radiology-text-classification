#!/bin/bash
NLP_ROOT="/proc_data1/bd5/nlp"
apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	setfit.sif \
	/nlp/src/setfit/optuna_head.py -c $1