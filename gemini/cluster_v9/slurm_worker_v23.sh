#!/bin/bash
#SBATCH --job-name=qmem_v23
#SBATCH --partition=submit,tier3,submit-gpu
#SBATCH --array=0-63
#SBATCH --cpus-per-task=8
#SBATCH --mem=18G
#SBATCH --time=02:00:00
#SBATCH --output=/work/submit/harshila/6s980_qmem_v9/logs_v23/slurm_%A_%a.out
#SBATCH --error=/work/submit/harshila/6s980_qmem_v9/logs_v23/slurm_%A_%a.err

set -u
WORK=/work/submit/harshila/6s980_qmem_v9
cd "$WORK"
SHARD="${SLURM_ARRAY_TASK_ID}"
OUT=$(printf "%s/results_v23/worker_%03d.txt" "$WORK" "$SHARD")
: > "$OUT"

grep -E "^${SHARD}\|" "$WORK/manifest_v23.txt" | while IFS='|' read -r sh tid mode cpath a1 a2 a3 a4 a5 a6 a7; do
    if [ "$mode" = "paired_readout" ]; then
        res=$("$WORK/kmc_engine_v19" paired_readout "$cpath" "$a1" "$a2" "$a3" "$a4" "$a5" "$a6" "$a7" 8 2>&1)
        rc=$?
    elif [ "$mode" = "iid_readout" ]; then
        res=$("$WORK/kmc_engine_v19" iid_readout "$cpath" "$a1" "$a2" "$a3" "$a4" "$a5" 8 2>&1)
        rc=$?
    else
        res="UNKNOWN_MODE"
        rc=1
    fi
    if [ $rc -eq 0 ]; then
        last=$(printf "%s\n" "$res" | tail -n 1)
        printf "%s | %s\n" "$tid" "$last" >> "$OUT"
    else
        printf "%s | ERROR rc=%d msg=%s\n" "$tid" "$rc" "$(printf "%s" "$res" | tr '\n' ' ')" >> "$OUT"
    fi
done
echo "ALL_DONE shard=${SHARD}" >> "$OUT"
