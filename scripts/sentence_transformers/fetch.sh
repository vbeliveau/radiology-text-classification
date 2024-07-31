#!/bin/bash
apptainer run \
	--nv \
	-H /proc_data1/bd5/nlp:/nlp \
	setfit.sif \
	python /nlp/src/sentence_transformers/fetch.py -c $1