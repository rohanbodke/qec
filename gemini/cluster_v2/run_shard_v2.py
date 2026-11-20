#!/usr/bin/env python3
import os, subprocess, sys, time

WORK_DIR = "/work/submit/harshila/6s980_qmem_v2"
DATA_DIR = os.path.join(WORK_DIR, "data")
RES_DIR = os.path.join(WORK_DIR, "results")
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v2")

sid = int(sys.argv[1])
nsh = int(sys.argv[2])

with open(os.path.join(WORK_DIR, "tasks_v2.txt")) as f:
    tasks = [l.strip() for l in f if l.strip()]

out_p = os.path.join(RES_DIR, f"shard_{sid:03d}.txt")
with open(out_p, "w") as out:
    for i, spec in enumerate(tasks):
        if (i % nsh) != sid:
            continue
        parts = spec.split("|")
        suite, label, mode = parts[0], parts[1], parts[2]
        if mode == "kmc_coupled":
            cmd = [ENGINE, mode, os.path.join(DATA_DIR, parts[3]), os.path.join(DATA_DIR, parts[4])] + parts[5:]
        else:
            cmd = [ENGINE, mode, os.path.join(DATA_DIR, parts[3])] + parts[4:]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            dt = time.time() - t0
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    rec = f"{suite}|{label}|dt={dt:.2f}s|{line}"
                    out.write(rec + "\n")
                    out.flush()
                    print(rec, flush=True)
            else:
                out.write(f"ERR|{suite}|{label}|rc={proc.returncode}|{proc.stderr.strip()}\n")
        except subprocess.TimeoutExpired:
            out.write(f"TIMEOUT|{suite}|{label}\n")
