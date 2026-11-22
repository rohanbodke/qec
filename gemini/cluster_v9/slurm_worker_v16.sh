#!/bin/bash
#SBATCH --job-name=qmem_v16
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:15:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs/v16_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs/v16_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v9
python3 /work/submit/harshila/6s980_qmem_v9/run_shard_v16.py "$SLURM_ARRAY_TASK_ID"
