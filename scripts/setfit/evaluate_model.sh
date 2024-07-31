#!/bin/bash

apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	setfit.sif \
	/nlp/src/setfit/evaluate_model.py \
	--config_json $1 \
	--tsne \
	"${@:2}"