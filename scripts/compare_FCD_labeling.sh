#!/bin/bash

apptainer run \
	--net --network none \
	-H /proc_data1/bd5/nlp:/nlp \
    -B /proc_data1/bd5/doccano:/doccano \
	nlp.sif \
	/nlp/src/compare_FCD_labeling.py \
    --doccano_dir /doccano \
	"$@"