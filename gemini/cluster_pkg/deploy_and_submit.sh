#!/bin/bash
set -euo pipefail
WORK_DIR="/work/submit/harshila/6s980_qmem"
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/data" "$WORK_DIR/results"
cd "$WORK_DIR"

echo "Compiling kmc_engine.cpp on SubMIT..."
g++ -O3 -march=x86-64 -std=c++20 -pthread "$WORK_DIR/kmc_engine.cpp" -o "$WORK_DIR/kmc_engine"

echo "Submitting Slurm build + array pipeline..."
PREP_JOB=$(sbatch --parsable \
  --job-name=qmem_prep \
  --partition=submit,tier3,submit-gpu-express \
  --cpus-per-task=4 \
  --mem=8G \
  --time=00:15:00 \
  --output="$WORK_DIR/logs/prep_%j.out" \
  --error="$WORK_DIR/logs/prep_%j.err" \
  --wrap="python3 $WORK_DIR/build_codes_on_cluster.py")

echo "PREP_JOB=$PREP_JOB"

ARR_JOB=$(sbatch --parsable \
  --dependency=afterok:$PREP_JOB \
  --array=0-47 \
  "$WORK_DIR/slurm_worker.sh")

echo "ARR_JOB=$ARR_JOB"
squeue -u harshila
