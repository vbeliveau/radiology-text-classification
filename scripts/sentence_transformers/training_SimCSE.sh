#!/bin/bash
apptainer run \
	--net --network none \
	--nv \
	-H /proc_data1/bd5/nlp:/nlp \
	setfit.sif \
	python /nlp/src/sentence_transformers_SimCSE/training.py -c $1