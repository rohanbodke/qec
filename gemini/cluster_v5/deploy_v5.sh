#!/bin/bash
set -euo pipefail
WORK_DIR="/work/submit/harshila/6s980_qmem_v5"
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/data" "$WORK_DIR/results"
cd "$WORK_DIR"

echo "Compiling kmc_engine_v5.cpp..."
g++ -O3 -march=x86-64 -std=c++20 -pthread "$WORK_DIR/kmc_engine_v5.cpp" -o "$WORK_DIR/kmc_engine_v5"

PREP=$(sbatch --parsable \
  --job-name=qmem_v5_prep \
  --partition=submit,submit-gpu,submit-gpu-express \
  --cpus-per-task=16 \
  --mem=20G \
  --time=00:25:00 \
  --output="$WORK_DIR/logs/prep_%j.out" \
  --error="$WORK_DIR/logs/prep_%j.err" \
  --wrap="python3 $WORK_DIR/build_v5_z3_and_2cycle.py")
echo "PREP_V5=$PREP"

ARR=$(sbatch --parsable \
  --dependency=afterok:$PREP \
  --array=0-63 \
  "$WORK_DIR/slurm_worker_v5.sh")
echo "ARR_V5=$ARR"
