#!/bin/bash
apptainer run \
	--net --network none \
	--nv \
	--bind `pwd`:/nlp -H `pwd`:/nlp \
	setfit.sif \
	/nlp/src/setfit/train_end_to_end.py -c $1