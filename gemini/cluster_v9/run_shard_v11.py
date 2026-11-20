#!/usr/bin/env python3
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v10")
MANIFEST = os.path.join(WORK_DIR, "manifest_v11.txt")
RES_DIR = os.path.join(WORK_DIR, "results_v11")


def main():
    shard_id = int(sys.argv[1])
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
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
            tid, mode = parts[0], parts[1]
            if mode == "kmc":
                _, _, cp, jp, dec, b, tr, mfl = parts
                cmd = [ENGINE, "kmc", cp, jp, dec, b, tr, mfl, str(threads)]
            elif mode == "rg_kmc":
                _, _, cp, dp, b, tr, mfl = parts
                cmd = [ENGINE, "rg_kmc", cp, dp, b, tr, mfl, str(threads)]
            elif mode == "rg_flow":
                _, _, rgp, b, teq, tr = parts
                cmd = [ENGINE, "rg_flow", rgp, b, teq, tr, str(threads)]
            elif mode == "cool":
                _, _, cp, jp, dec, proto, init_m, b0, b1, tt, tr = parts
                cmd = [ENGINE, "cool", cp, jp, dec, proto, init_m, b0, b1, tt, tr, str(threads)]
            else:
                out.write(f"{tid} | UNKNOWN_MODE {mode}\n")
                out.flush()
                continue
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=540, check=True)
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
