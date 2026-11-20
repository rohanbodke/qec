#!/bin/bash
set -euo pipefail
WORK_DIR="/work/submit/harshila/6s980_qmem_v4"
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/data" "$WORK_DIR/results"
cd "$WORK_DIR"

echo "Compiling kmc_engine_v4.cpp..."
g++ -O3 -march=x86-64 -std=c++20 -pthread "$WORK_DIR/kmc_engine_v4.cpp" -o "$WORK_DIR/kmc_engine_v4"

PREP=$(sbatch --parsable \
  --job-name=qmem_v4_prep \
  --partition=submit,submit-gpu,submit-gpu-express \
  --cpus-per-task=4 \
  --mem=12G \
  --time=00:20:00 \
  --output="$WORK_DIR/logs/prep_%j.out" \
  --error="$WORK_DIR/logs/prep_%j.err" \
  --wrap="python3 $WORK_DIR/build_v4_scaling.py")
echo "PREP_V4=$PREP"

ARR=$(sbatch --parsable \
  --dependency=afterok:$PREP \
  --array=0-63 \
  "$WORK_DIR/slurm_worker_v4.sh")
echo "ARR_V4=$ARR"
