#!/bin/bash

apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	transformers.sif \
	/nlp/src/transformers/evaluate.py -c $1