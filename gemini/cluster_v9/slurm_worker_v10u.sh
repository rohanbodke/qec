#!/bin/bash
#SBATCH --job-name=qmem_v10u
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs/v10u_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs/v10u_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v9
python3 /work/submit/harshila/6s980_qmem_v9/run_shard_v10u.py "$SLURM_ARRAY_TASK_ID"
