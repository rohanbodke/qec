#!/bin/bash
#SBATCH --job-name=qmem_v19
#SBATCH --partition=tier3,submit
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=01:15:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs/v19_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs/v19_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v9
python3 /work/submit/harshila/6s980_qmem_v9/run_shard_v19.py "$SLURM_ARRAY_TASK_ID" 8
