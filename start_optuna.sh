#!/bin/bash
apptainer run \
	--bind `pwd`:/nlp -H `pwd`:/nlp \
	setfit.sif \
	optuna-dashboard sqlite:////nlp/$1