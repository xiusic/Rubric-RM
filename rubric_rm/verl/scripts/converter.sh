#!/bin/bash

RUN_DIR=$1

for GLOBAL_STEP in $(ls -d "$RUN_DIR"/global_step_* | sort -V); do
    MODEL_TYPE=$GLOBAL_STEP/actor
    python rubric_rm/verl/scripts/converter.py --local_dir $MODEL_TYPE
    rm -rf $MODEL_TYPE/*pt
    rm -rf $GLOBAL_STEP/data.pt
    mv $MODEL_TYPE/huggingface/* $GLOBAL_STEP
    rm -rf $MODEL_TYPE
done
