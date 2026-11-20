#!/usr/bin/env python3
import os
import subprocess
import sys
import time

WORK_DIR = "/work/submit/harshila/6s980_qmem_v5"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v5")

sid = int(sys.argv[1])
nsh = int(sys.argv[2])

with open(os.path.join(WORK_DIR, "tasks_v5.txt")) as f:
    tasks = [l.strip() for l in f if l.strip()]

out_p = os.path.join(RES_DIR, f"shard_{sid:03d}.txt")
with open(out_p, "w") as out:
    for i, spec in enumerate(tasks):
        if (i % nsh) != sid:
            continue
        parts = spec.split("|")
        suite, label, mode = parts[0], parts[1], parts[2]
        cpath = os.path.join(DATA_DIR, parts[3])
        if mode == "kmc":
            jspec = parts[4] if parts[4] == "uniform" else os.path.join(DATA_DIR, parts[4])
            cmd = [ENGINE, mode, cpath, jspec] + parts[5:]
        else:
            cmd = [ENGINE, mode, cpath] + parts[4:]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
            dt = time.time() - t0
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    rec = f"{suite}|{label}|dt={dt:.2f}s|{line}"
                    out.write(rec + "\n")
                    out.flush()
                    print(rec, flush=True)
            else:
                out.write(f"ERR|{suite}|{label}|rc={proc.returncode}|{proc.stderr.strip()}\n")
                out.flush()
        except subprocess.TimeoutExpired:
            out.write(f"TIMEOUT|{suite}|{label}\n")
            out.flush()
