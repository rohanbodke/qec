#!/bin/bash
set -euo pipefail
WORK_DIR="/work/submit/harshila/6s980_qmem_v6"
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/data" "$WORK_DIR/results" "$WORK_DIR/movies"
cd "$WORK_DIR"

echo "Compiling kmc_engine_v6.cpp..."
g++ -O3 -march=x86-64 -std=c++20 -pthread "$WORK_DIR/kmc_engine_v6.cpp" -o "$WORK_DIR/kmc_engine_v6"

PREP=$(sbatch --parsable \
  --job-name=qmem_v6_prep \
  --partition=submit \
  --cpus-per-task=16 \
  --mem=28G \
  --time=00:35:00 \
  --output="$WORK_DIR/logs/prep_%j.out" \
  --error="$WORK_DIR/logs/prep_%j.err" \
  --wrap="python3 $WORK_DIR/build_v6_ambitious.py")
echo "PREP_V6=$PREP"

ARR=$(sbatch --parsable \
  --dependency=afterok:$PREP \
  --array=0-95 \
  "$WORK_DIR/slurm_worker_v6.sh")
echo "ARR_V6=$ARR"
