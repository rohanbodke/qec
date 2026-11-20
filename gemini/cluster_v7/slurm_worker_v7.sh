#!/bin/bash
#SBATCH --job-name=qmem_v7
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v7/logs/arr_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v7/logs/arr_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v7
python3 /work/submit/harshila/6s980_qmem_v7/run_shard_v7.py "$SLURM_ARRAY_TASK_ID" 24
