#!/bin/bash
#SBATCH --job-name=qmem_v7_bld
#SBATCH --partition=submit
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v7/logs/build_calib_%j.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v7/logs/build_calib_%j.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v7

echo "=== STEP 1: BUILD & VERIFY ALL V7 CODES ==="
python3 /work/submit/harshila/6s980_qmem_v7/build_v7_rescue.py

echo ""
echo "=== STEP 2: HEAD-TO-HEAD KMC vs V7 EXACT FFS CALIBRATION ==="
ENG=/work/submit/harshila/6s980_qmem_v7/kmc_engine_v7
D=/work/submit/harshila/6s980_qmem_v7/data

echo "[1] cubic_L5_X_err beta=5.00 (V4/V6 KMC ~ 9.4e8; V6 broken FFS was 1.23e8):"
$ENG ffs $D/cubic_L5_X_err.code uniform uf 5.00 2000 6 16
echo "[1b] cubic_L5_X_err beta=6.00 & 7.00 (check Deff == 6!):"
$ENG ffs $D/cubic_L5_X_err.code uniform uf 6.00 2000 6 16
$ENG ffs $D/cubic_L5_X_err.code uniform uf 7.00 2000 6 16

echo "[2] rottor_L2r1_s2_X_err beta=6.00 & 6.50:"
$ENG kmc $D/rottor_L2r1_s2_X_err.code uniform uf 6.00 64 100000000 16
$ENG ffs $D/rottor_L2r1_s2_X_err.code uniform uf 6.00 2200 6 16
$ENG kmc $D/rottor_L2r1_s2_X_err.code uniform uf 6.50 64 100000000 16
$ENG ffs $D/rottor_L2r1_s2_X_err.code uniform uf 6.50 2200 6 16

echo "[3] rottor_L2r1_s2_Z_err beta=6.00:"
$ENG kmc $D/rottor_L2r1_s2_Z_err.code uniform uf 6.00 48 120000000 16
$ENG ffs $D/rottor_L2r1_s2_Z_err.code uniform uf 6.00 2200 6 16

echo "[4] sq_L3r2_s2_X_err beta=6.50:"
$ENG kmc $D/sq_L3r2_s2_X_err.code uniform uf 6.50 48 100000000 16
$ENG ffs $D/sq_L3r2_s2_X_err.code uniform uf 6.50 2200 6 16

echo "[5] Commuting pruned rottor_cb3rib s=2,3,4 exact collapse (t_next=0.0):"
$ENG kmc $D/rottor_cb3rib_s2_Z_err.code uniform uf 6.50 64 1000000 16
$ENG kmc $D/rottor_cb3rib_s3_Z_err.code uniform uf 6.50 64 1000000 16
$ENG kmc $D/rottor_cb3rib_s4_Z_err.code uniform uf 6.50 64 1000000 16

echo "[6] Sub-threshold 3D quantum codeword self-healing (rottor_L2r1_s2, p0=0.02 vs 0.12):"
$ENG cool $D/rottor_L2r1_s2_X_err.code uniform uf quench noisy_0.02 6.5 6.5 1000.0 100 16
$ENG cool $D/rottor_L2r1_s2_X_err.code uniform uf quench noisy_0.04 6.5 6.5 1000.0 100 16
$ENG cool $D/rottor_L2r1_s2_X_err.code uniform uf quench noisy_0.12 6.5 6.5 1000.0 100 16

echo "=== CALIBRATION COMPLETE ==="
