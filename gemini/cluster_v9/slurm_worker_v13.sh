#!/bin/bash
#SBATCH --job-name=qmem_v13
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:35:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs/v13_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs/v13_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v9
python3 /work/submit/harshila/6s980_qmem_v9/run_shard_v13.py "$SLURM_ARRAY_TASK_ID"
