#!/bin/sh
source venv/bin/activate # activate training environment

set -x

JOB_NAME=$1
CONFIG=$2
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:3}
GPUS=${GPUS:-2}

srun --job-name=${JOB_NAME} \
     --output=output/jobs/%J_%x.out \
     --gres=gpu:2 \
     --time=2-00:00:00 \
     --ntasks-per-node=2 \
     --cpus-per-task=12 \
     --mem-per-gpu=60G \
     --partition=gpu \
     --ntasks=${GPUS} \
     ${SRUN_ARGS} \
     python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS} &
