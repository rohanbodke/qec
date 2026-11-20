#!/bin/bash
#SBATCH --job-name=qmem_v6
#SBATCH --partition=submit
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:55:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v6/logs/arr_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v6/logs/arr_%A_%a.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v6
python3 /work/submit/harshila/6s980_qmem_v6/run_shard_v6.py "$SLURM_ARRAY_TASK_ID" 96
