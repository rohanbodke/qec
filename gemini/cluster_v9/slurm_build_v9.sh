#!/bin/bash
#SBATCH --job-name=qmem_v9_bld
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs/build_%j.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs/build_%j.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v9
g++ -O3 -std=c++20 -pthread kmc_engine_v9.cpp -o kmc_engine_v9
python3 build_v9_quarter_million.py
