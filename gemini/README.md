# Simulating Three-Dimensional Self-Correcting Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the code constructors, C++20 thermal Monte Carlo and Forward Flux Sampling (FFS) simulation engines, MIT SubMIT Slurm orchestration scripts, raw benchmark logs (`V1`–`V7`), figure-rendering scripts, and compiled APS REVTeX 4.2 manuscripts investigating three-dimensional passive self-correcting quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Directory Layout

* **`paper/main.tex` & `paper/main.pdf`** — 6-page two-column APS REVTeX 4.2 manuscript (*Simulating Three-Dimensional Self-Correcting Quantum Memories: Compact Welded Cells, Cubic-Grid Limits, and Thermal Cooling*).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — 2-page APS REVTeX 4.2 summary / proposal.
* **`cluster_v7/`** — Final verified C++20 simulation suite and CSS code builder:
  * `kmc_engine_v7.cpp` — Rejection-free continuous-time Bortz–Kalos–Lebowitz (BKL) Glauber Kinetic Monte Carlo, 6-replica Forward Flux Sampling with exact ground-codeword committor relaxation, Union-Find cluster decoding, and zero-measurement storage self-healing / hot-state cooling benchmarks up to `56,312` qubits.
  * `build_v7_rescue.py` — Exact CSS welded code builder with local corner check-weight reduction (`w <= 6-7`), strict commutation verification (`H_X H_Z^T == 0`), pruned 3-arm T-bridge buttonhole tests, cubic-grid wire-stretching embeddings, `2 -> 1` sector pre-balancing, Haah's 3D Cubic Code, 3D Toric Code, and published Gu–Caha Random Layer Codes.
  * `run_shard_v7.py`, `slurm_build_and_calib_v7.sh`, `slurm_worker_v7.sh` — MIT SubMIT Slurm execution pipeline.
* **`src/`** — Analysis scripts, finite-cell verifiers, result parsers, and clean SVG/PNG figure renderers:
  * `render_v9_clear_figures.py` — Generates the 6 publication figures (`figures/fig_v9_1_classes.png` through `figures/fig_v9_6_cooling.png`).
  * `parse_v7_results.py` — Parses `cluster_v7_results.txt` into `parsed_v7_summary.txt`.
  * `exact_rib_cleaning_theorem.py`, `test_sheaf_exact.py`, `test_classical_bdl.py`, `analyze_seeds_and_ribs.py` — Exact local cell-cleaning, CSS sheaf, and 2D classical dual-welded verifiers.
* **`cluster_pkg/`, `cluster_v2/`–`cluster_v6/`** — Earlier experimental iterations and benchmark logs (`cluster_*_results.txt`, `parsed_*_summary.txt`).
* **`figures/`** — Rendered SVG and high-DPI PNG figures (`fig_v9_1_classes.png`–`fig_v9_6_cooling.png` used in the final manuscripts, plus historical figure iterations).
* **`data/`** — Exported small/medium `.code` benchmark instances.

---

## Reproducing Figures and Manuscripts

### 1. Parse Cluster Results & Render Clean `V9` Figures
```bash
python3 src/parse_v7_results.py
python3 src/render_v9_clear_figures.py
```

### 2. Compile the APS REVTeX 4.2 Manuscripts
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```

### 3. Re-run HPC Benchmarks on MIT SubMIT (`submit.mit.edu`)
```bash
scp -r cluster_v7/* submit.mit.edu:/work/submit/$USER/6s980_qmem_v7/
ssh submit.mit.edu "cd /work/submit/$USER/6s980_qmem_v7 && sbatch slurm_build_and_calib_v7.sh"
```
