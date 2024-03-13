#!/bin/bash
apptainer run \
	--net --network none \
	--nv \
	-H ${NLP_ROOT}:/nlp \
	setfit.sif \
	/nlp/src/setfit/evaluate_end_to_end.py --project_name $1 --tsne