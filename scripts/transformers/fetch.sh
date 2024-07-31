#!/bin/bash

apptainer run \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	transformers.sif \
	/nlp/src/transformers/fetch.py -c $1