#!/usr/bin/env python3
"""
Executes shard `shard_id` out of `num_shards` from `tasks.txt` on a Slurm compute node.
"""
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENGINE = os.path.join(BASE_DIR, "kmc_engine")

shard_id = int(sys.argv[1])
num_shards = int(sys.argv[2])

with open(os.path.join(BASE_DIR, "tasks.txt")) as f:
    all_tasks = [line.strip() for line in f if line.strip()]

my_tasks = [t for i, t in enumerate(all_tasks) if (i % num_shards) == shard_id]
out_file = os.path.join(RESULTS_DIR, f"shard_{shard_id:03d}.txt")

with open(out_file, "w") as out:
    for idx, spec in enumerate(my_tasks):
        parts = spec.split("|")
        suite, label, mode = parts[0], parts[1], parts[2]
        code_path = os.path.join(DATA_DIR, parts[3])
        args = [ENGINE, mode, code_path] + parts[4:]
        t0 = time.time()
        try:
            # Timeout each individual sub-task at 60s so no single point blocks a shard
            proc = subprocess.run(args, capture_output=True, text=True, timeout=75)
            dt = time.time() - t0
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    rec = f"{suite}|{label}|dt={dt:.2f}s|{line}"
                    out.write(rec + "\n")
                    out.flush()
                    print(rec, flush=True)
            else:
                print(f"ERR|{suite}|{label}|rc={proc.returncode}|{proc.stderr.strip()}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT|{suite}|{label}", flush=True)
