#!/bin/bash
#SBATCH --job-name=qmem_v4
#SBATCH --partition=submit,submit-gpu,submit-gpu-express
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:55:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v4/logs/arr_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v4/logs/arr_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v4
python3 /work/submit/harshila/6s980_qmem_v4/run_shard_v4.py "$SLURM_ARRAY_TASK_ID" 64
