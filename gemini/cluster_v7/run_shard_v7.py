#!/usr/bin/env python3
import os
import subprocess
import sys

WORK_DIR = "/work/submit/harshila/6s980_qmem_v7"
ENGINE = os.path.join(WORK_DIR, "kmc_engine_v7")
MANIFEST = os.path.join(WORK_DIR, "manifest_v7.txt")
RES_DIR = os.path.join(WORK_DIR, "results")


def main():
    shard_id = int(sys.argv[1])
    num_shards = int(sys.argv[2])
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    with open(MANIFEST) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    my_lines = [ln for idx, ln in enumerate(lines) if idx % num_shards == shard_id]
    out_file = os.path.join(RES_DIR, f"shard_{shard_id:03d}.txt")

    with open(out_file, "w") as out:
        for ln in my_lines:
            parts = ln.split("|")
            tid, mode = parts[0], parts[1]
            if mode == "kmc":
                _, _, code_p, j_p, dec, beta, trials, max_fl = parts
                cmd = [ENGINE, "kmc", code_p, j_p, dec, beta, trials, max_fl, str(threads)]
            elif mode == "ffs":
                _, _, code_p, j_p, dec, beta, shots, reps, mov = parts
                cmd = [ENGINE, "ffs", code_p, j_p, dec, beta, shots, reps, str(threads)]
                if mov:
                    cmd.append(mov)
            elif mode == "cool":
                _, _, code_p, j_s, dec, proto, init_m, bi, bf, ttot, trials = parts
                cmd = [ENGINE, "cool", code_p, j_s, dec, proto, init_m, bi, bf, ttot, trials, str(threads)]
            else:
                continue

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=True)
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
