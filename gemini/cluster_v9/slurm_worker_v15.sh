#!/bin/bash
#SBATCH --job-name=qmem_v15
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:50:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs/v15_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs/v15_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v9
python3 /work/submit/harshila/6s980_qmem_v9/run_shard_v15.py "$SLURM_ARRAY_TASK_ID"
