#!/bin/bash
ROOT_DIR="/proc_data1/bd5/nlp"
apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	setfit.sif \
	/nlp/src/setfit/evaluate_model.py \
	--config_json $1 \
	--tsne \
	"${@:2}"