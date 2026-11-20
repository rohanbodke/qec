#!/bin/bash
#SBATCH --job-name=qmem_v9d
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=00:13:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs/v9d_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs/v9d_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v9
python3 /work/submit/harshila/6s980_qmem_v9/run_shard_v9d.py "$SLURM_ARRAY_TASK_ID"
