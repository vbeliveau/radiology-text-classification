#!/bin/bash
NLP_ROOT="/proc_data1/bd5/nlp"
apptainer run \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	setfit.sif \
	/nlp/src/setfit/fetch.py -c $1