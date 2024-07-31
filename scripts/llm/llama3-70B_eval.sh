#!/bin/bash
NLP_ROOT="/proc_data1/bd5/nlp"

# apptainer run \
#	--net --network none \
#	--nv \
#	-H ${NLP_ROOT}:/nlp \
#	llm.sif \
#	/nlp/src/llm/eval.py \
#	--model $1 \
#	--project $2 \
#	"${@:3}"
	
bash ${NLP_ROOT}/scripts/llm/eval.sh llama3-70B FCD \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10

bash ${NLP_ROOT}/scripts/llm/eval.sh llama3-70B MTS \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10

bash ${NLP_ROOT}/scripts/llm/eval.sh llama3-70B hippocampus \
    --data_dir /nlp/data/preproc-melba-translated \
    --n_samples 10