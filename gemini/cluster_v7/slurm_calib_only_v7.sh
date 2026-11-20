#!/bin/bash
#SBATCH --job-name=qmem_v7_cal
#SBATCH --partition=submit
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v7/logs/calib_fast_%j.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v7/logs/calib_fast_%j.err

set -euo pipefail
cd /work/submit/harshila/6s980_qmem_v7

echo "=== FAST ZOBRIST-CACHED BOUNDED-LAMBDA V7 CALIBRATION ==="
ENG=/work/submit/harshila/6s980_qmem_v7/kmc_engine_v7
D=/work/submit/harshila/6s980_qmem_v7/data

echo "[1] cubic_L5_X_err beta=5.00, 6.00, 7.00 (V4/V6 KMC b=5 ~ 9.4e8; check Deff == 6!):"
$ENG ffs $D/cubic_L5_X_err.code uniform uf 5.00 2000 6 16
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

echo "[6] Sub-threshold 3D quantum codeword self-healing (rottor_L2r1_s2, p0=0.02, 0.04, 0.12):"
$ENG cool $D/rottor_L2r1_s2_X_err.code uniform uf quench noisy_0.02 6.5 6.5 1000.0 100 16
$ENG cool $D/rottor_L2r1_s2_X_err.code uniform uf quench noisy_0.04 6.5 6.5 1000.0 100 16
$ENG cool $D/rottor_L2r1_s2_X_err.code uniform uf quench noisy_0.12 6.5 6.5 1000.0 100 16

echo "=== FAST CALIBRATION COMPLETE ==="
