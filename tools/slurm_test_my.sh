#!/bin/sh
source venv/bin/activate # activate training environment

set -x

JOB_NAME=$1
CONFIG=$2
CHECKPOINT=$3
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}
GPUS=${GPUS:-2}

srun --job-name=${JOB_NAME} \
     --output=output/jobs/%J_%x.out \
     --gres=gpu:2 \
     --time=1-00:00:00 \
     --ntasks-per-node=1 \
     --cpus-per-task=24 \
     --mem-per-gpu=60G \
     --partition=gpu \
     --ntasks=1 \
     ${SRUN_ARGS} \
     tools/echorun.sh python -u tools/test.py ${CONFIG} ${CHECKPOINT} ${PY_ARGS} &
