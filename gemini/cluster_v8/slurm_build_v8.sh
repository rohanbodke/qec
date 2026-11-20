#!/bin/bash
#SBATCH --job-name=qmem_v8_bld
#SBATCH --partition=submit,tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v8/logs/build_%j.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v8/logs/build_%j.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v8
g++ -O3 -std=c++20 -pthread kmc_engine_v8.cpp -o kmc_engine_v8
g++ -O3 -std=c++20 prove_exact_soundness.cpp -o prove_exact_soundness
./prove_exact_soundness > exact_soundness_v8_report.txt
cat exact_soundness_v8_report.txt
python3 build_v8_rigorous.py
