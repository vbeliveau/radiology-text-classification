#!/bin/bash

# For debugging text_utils
# --bind /proc_data1/bd5/text_utils/src/text_utils/text_utils.py:/venv/lib/python3.10/site-packages/text_utils/text_utils.py \

apptainer run \
	--net --network none \
	-H /proc_data1/bd5/nlp:/nlp \
    -B /proc_data1/bd5/doccano:/doccano \
	nlp.sif \
	/nlp/src/compare_FCD_labeling.py \
    --doccano_dir /doccano \
	"$@"