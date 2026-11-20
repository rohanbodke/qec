#!/bin/bash
set -euo pipefail
WORK_DIR="/work/submit/harshila/6s980_qmem_v3"
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/data" "$WORK_DIR/results"
cd "$WORK_DIR"

echo "Compiling kmc_engine_v3.cpp..."
g++ -O3 -march=x86-64 -std=c++20 -pthread "$WORK_DIR/kmc_engine_v3.cpp" -o "$WORK_DIR/kmc_engine_v3"

PREP=$(sbatch --parsable \
  --job-name=qmem_v3_prep \
  --partition=submit,submit-gpu,submit-gpu-express \
  --cpus-per-task=4 \
  --mem=8G \
  --time=00:15:00 \
  --output="$WORK_DIR/logs/prep_%j.out" \
  --error="$WORK_DIR/logs/prep_%j.err" \
  --wrap="python3 $WORK_DIR/build_v3_and_tasks.py")
echo "PREP_V3=$PREP"

ARR=$(sbatch --parsable \
  --dependency=afterok:$PREP \
  --array=0-55 \
  "$WORK_DIR/slurm_worker_v3.sh")
echo "ARR_V3=$ARR"
