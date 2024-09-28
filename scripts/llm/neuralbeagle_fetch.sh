#!/bin/bash

apptainer run \
	-H ${NLP_ROOT}:/nlp \
	llm.sif \
	/nlp/src/llm/neuralbeagle_fetch.py