#!/bin/bash
#SBATCH --job-name=qmem_v3
#SBATCH --partition=submit,submit-gpu,submit-gpu-express
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v3/logs/arr_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v3/logs/arr_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v3
python3 /work/submit/harshila/6s980_qmem_v3/run_shard_v3.py "$SLURM_ARRAY_TASK_ID" 56
