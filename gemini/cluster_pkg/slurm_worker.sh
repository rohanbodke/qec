#!/bin/bash
# Slurm array worker: executes shard $SLURM_ARRAY_TASK_ID out of $NUM_SHARDS from tasks.txt
#SBATCH --job-name=qmem_6s980
#SBATCH --partition=submit,tier3,submit-gpu,submit-gpu-express
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/work/submit/harshila/6s980_qmem/logs/arr_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem/logs/arr_%A_%a.err

set -euo pipefail
WORK_DIR="/work/submit/harshila/6s980_qmem"
cd "$WORK_DIR"

python3 "$WORK_DIR/run_shard.py" "$SLURM_ARRAY_TASK_ID" 48
