#!/usr/bin/env python3
import glob
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v19")
RES_DIR = os.path.join(WORK_DIR, "results_v19")


def load_done_tids():
    done = set()
    for path in glob.glob(os.path.join(RES_DIR, "*.txt")):
        try:
            with open(path) as f:
                for raw in f:
                    if "|" in raw and ("PAIRED_READOUT" in raw or "IID_READOUT" in raw):
                        tid = raw.split("|", 1)[0].strip()
                        done.add(tid)
        except OSError:
            pass
    return done


def main():
    shard_id = int(sys.argv[1])
    threads = int(sys.argv[2])
    manifest_p = os.path.join(WORK_DIR, "manifest_v19.txt")

    os.makedirs(RES_DIR, exist_ok=True)
    done = load_done_tids()

    my_lines = []
    with open(manifest_p) as f:
        for raw in f:
            if not raw.strip():
                continue
            sh, rest = raw.strip().split("|", 1)
            if int(sh) == shard_id:
                my_lines.append(rest)

    out_p = os.path.join(RES_DIR, f"worker_{shard_id:03d}.txt")
    with open(out_p, "a") as out:
        for ln in my_lines:
            parts = ln.split("|")
            tid = parts[0]
            if tid in done:
                continue
            args = parts[1:]
            cmd = [ENGINE] + args + [str(threads)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=2700, check=True)
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
