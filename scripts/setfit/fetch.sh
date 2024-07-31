#!/bin/bash

apptainer run \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	setfit.sif \
	/nlp/src/setfit/fetch.py -c $1