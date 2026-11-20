#!/usr/bin/env python3
import os, subprocess, sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v9d")
MANIFEST = os.path.join(WORK_DIR, "manifest_v10u.txt")
RES_DIR = os.path.join(WORK_DIR, "results_v10u")

def main():
    shard_id = int(sys.argv[1])
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    os.makedirs(RES_DIR, exist_ok=True)
    my_lines = []
    with open(MANIFEST) as f:
        for raw in f:
            if not raw.strip(): continue
            sh, rest = raw.strip().split("|", 1)
            if int(sh) == shard_id: my_lines.append(rest)
    with open(os.path.join(RES_DIR, f"shard_{shard_id:03d}.txt"), "w") as out:
        for ln in my_lines:
            tid, _, cp, jp, dec, b, tr, mfl = ln.split("|")
            try:
                r = subprocess.run([ENGINE, "kmc", cp, jp, dec, b, tr, mfl, str(threads)],
                                   capture_output=True, text=True, timeout=480, check=True)
                out.write(f"{tid} | {r.stdout.strip()}\n"); out.flush()
            except subprocess.TimeoutExpired:
                out.write(f"{tid} | TIMEOUT\n"); out.flush()
            except Exception as e:
                out.write(f"{tid} | ERROR {e}\n"); out.flush()

if __name__ == "__main__":
    main()
