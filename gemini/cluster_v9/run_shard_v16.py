#!/usr/bin/env python3
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v16")
MANIFEST = os.path.join(WORK_DIR, "manifest_v16.txt")
RES_DIR = os.path.join(WORK_DIR, "results_v16")


def main():
    shard_id = int(sys.argv[1])
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    os.makedirs(RES_DIR, exist_ok=True)
    my_lines = []
    with open(MANIFEST) as f:
        for raw in f:
            if not raw.strip():
                continue
            sh, rest = raw.strip().split("|", 1)
            if int(sh) == shard_id:
                my_lines.append(rest)

    out_p = os.path.join(RES_DIR, f"shard_{shard_id:03d}.txt")
    with open(out_p, "w") as out:
        for ln in my_lines:
            parts = ln.split("|")
            tid = parts[0]
            args = parts[1:]
            cmd = [ENGINE] + args + [str(threads)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, check=True)
                out.write(f"{tid} | {r.stdout.strip()}\n")
                out.flush()
            except subprocess.TimeoutExpired:
                out.write(f"{tid} | TIMEOUT\n")
                out.flush()
            except Exception as e:
                err_s = getattr(e, "stderr", "") or str(e)
                out.write(f"{tid} | ERROR {err_s.strip()}\n")
                out.flush()


if __name__ == "__main__":
    main()
