#!/bin/bash
apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	transformers.sif \
	/nlp/src/transformers/train_classifier_head.py -c $1