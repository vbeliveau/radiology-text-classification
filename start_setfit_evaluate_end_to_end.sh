#!/bin/bash
apptainer run \
	--net --network none \
	--nv \
	--bind `pwd`:/nlp -H `pwd`:/nlp \
	setfit.sif \
	/nlp/src/setfit/evaluate_end_to_end.py --project_name $1 --tsne