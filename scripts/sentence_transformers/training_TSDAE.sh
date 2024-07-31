#!/bin/bash
apptainer run \
	--net --network none \
	--nv \
	-H /proc_data1/bd5/nlp:/nlp \
    --env NLTK_DATA=/root/nltk_data \
    	--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	setfit.sif \
	python /nlp/src/sentence_transformers/training_TSDAE.py -c $1