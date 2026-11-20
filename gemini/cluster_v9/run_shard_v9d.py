#!/usr/bin/env python3
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v9d")
MANIFEST = os.path.join(WORK_DIR, "manifest_v9d.txt")
RES_DIR = os.path.join(WORK_DIR, "results_v9d")


def main():
    shard_id = int(sys.argv[1])
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "3"))
    os.makedirs(RES_DIR, exist_ok=True)

    my_lines = []
    with open(MANIFEST) as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            sh_str, rest = ln.split("|", 1)
            if int(sh_str) == shard_id:
                my_lines.append(rest)

    out_file = os.path.join(RES_DIR, f"shard_{shard_id:03d}.txt")
    with open(out_file, "w") as out:
        for ln in my_lines:
            parts = ln.split("|")
            tid, mode = parts[0], parts[1]
            if mode == "kmc":
                _, _, code_p, j_p, dec, beta, trials, max_fl = parts
                cmd = [ENGINE, "kmc", code_p, j_p, dec, beta, trials, max_fl, str(threads)]
            else:
                continue

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
                line_out = res.stdout.strip().replace("\n", " ; ")
                out.write(f"{tid} | {line_out}\n")
                out.flush()
            except subprocess.TimeoutExpired:
                out.write(f"{tid} | TIMEOUT\n")
                out.flush()
            except Exception as e:
                out.write(f"{tid} | ERROR {e}\n")
                out.flush()


if __name__ == "__main__":
    main()
