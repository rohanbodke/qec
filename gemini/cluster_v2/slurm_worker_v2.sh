#!/bin/bash
#SBATCH --job-name=qmem_v2
#SBATCH --partition=submit,submit-gpu,submit-gpu-express
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v2/logs/arr_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v2/logs/arr_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v2
python3 /work/submit/harshila/6s980_qmem_v2/run_shard_v2.py "$SLURM_ARRAY_TASK_ID" 56
