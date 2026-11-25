import os

WORK_DIR = "/work/submit/harshila/6s980_qmem_v9"
D19 = os.path.join(WORK_DIR, "data_v19")
D20 = os.path.join(WORK_DIR, "data_v20")

# Pure standard J = 1 runs (exact physical beta, zero rescaling!)
arms = [
    # 2D Toric L=16 at beta = 4.50, 5.00
    (D20, "v20_tor2d_L16",       "br4.50", 4.50, 0.5, 3.5, 31, 96, 20000000),
    (D20, "v20_tor2d_L16",       "br5.00", 5.00, 0.5, 3.8, 34, 96, 20000000),
    # 1-cycle 3D Welded L=4 at beta = 4.50, 5.00, 5.50, 6.00, 7.00
    (D20, "v20_mich1c_L4",       "br4.50", 4.50, 0.8, 3.8, 31, 96, 20000000),
    (D20, "v20_mich1c_L4",       "br5.00", 5.00, 1.0, 4.0, 31, 96, 20000000),
    (D20, "v20_mich1c_L4",       "br5.50", 5.50, 1.2, 4.4, 33, 96, 30000000),
    (D20, "v20_mich1c_L4",       "br6.00", 6.00, 1.5, 4.8, 34, 96, 30000000),
    (D20, "v20_mich1c_L4",       "br7.00", 7.00, 2.0, 5.4, 35, 96, 40000000),
    # 3D-packable Alternating Welded C3 C2 (N=107,140) at beta = 4.50, 5.00, 5.50, 6.00
    (D19, "v19_sq_C3_C2",        "br4.50", 4.50, 1.0, 4.0, 25, 64, 40000000),
    (D19, "v19_sq_C3_C2",        "br5.00", 5.00, 1.4, 4.6, 25, 64, 50000000),
    (D19, "v19_sq_C3_C2",        "br5.50", 5.50, 1.8, 5.2, 25, 64, 60000000),
    (D19, "v19_sq_C3_C2",        "br6.00", 6.00, 2.5, 6.0, 29, 64, 80000000),
    # 3D-packable Symmetric Welded C3 C2^ZX (N=87,652) at beta = 4.50, 5.00, 5.50
    (D20, "v20_weld_4s_C3C2ZX",  "br4.50", 4.50, 1.2, 4.4, 25, 64, 40000000),
    (D20, "v20_weld_4s_C3C2ZX",  "br5.00", 5.00, 1.8, 5.2, 25, 64, 50000000),
    (D20, "v20_weld_4s_C3C2ZX",  "br5.50", 5.50, 2.5, 6.0, 29, 64, 80000000),
    # Haah 3D Cubic Code L=7 (N=686, Z^3 local) at J=1, beta = 4.00, 4.25, 4.50, 5.00
    (D20, "v20_haah_L7",         "br4.00", 4.00, 3.5, 7.2, 30, 96, 60000000),
    (D20, "v20_haah_L7",         "br4.25", 4.25, 4.0, 7.8, 31, 96, 80000000),
    (D20, "v20_haah_L7",         "br4.50", 4.50, 4.5, 8.4, 27, 96, 100000000),
    (D20, "v20_haah_L7",         "br5.00", 5.00, 5.0, 8.8, 26, 64, 120000000),
    # Haah 3D Cubic size sweep at J=1, beta = 4.00 across L = 5, 9, 11, 13, 15, 17
    (D20, "v20_haah_L5",         "br4.00", 4.00, 3.5, 7.2, 30, 96, 60000000),
    (D20, "v20_haah_L9",         "br4.00", 4.00, 3.5, 7.0, 29, 96, 80000000),
    (D20, "v20_haah_L11",        "br4.00", 4.00, 3.5, 7.0, 29, 96, 80000000),
    (D20, "v20_haah_L13",        "br4.00", 4.00, 3.5, 6.8, 27, 96, 100000000),
    (D20, "v20_haah_L15",        "br4.00", 4.00, 3.2, 6.5, 27, 96, 100000000),
    (D20, "v20_haah_L17",        "br4.00", 4.00, 3.2, 6.6, 28, 96, 120000000),
]

lines = []
sh = 0
for ddir, tag, mlbl, bval, lg0, lg1, ng, tot, maxf in arms:
    for sec in ("X_err", "Z_err"):
        cp = os.path.join(ddir, f"{tag}_{sec}.code")
        for off in range(0, tot, 32):
            ntr = min(32, tot - off)
            tid = f"v20__{tag}_{sec}__{mlbl}__b{bval:.5f}__o{off}"
            lines.append(f"{sh}|{tid}|paired_readout|{cp}|{bval:.6f}|{ntr}|{off+1}|{lg0:.4f}|{lg1:.4f}|{ng}|{maxf}\n")
            sh = (sh + 1) % 48

with open("manifest_v20.txt", "w") as f:
    f.writelines(lines)
print("Wrote", len(lines), "tasks across 48 shards")
