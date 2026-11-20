# Simulating Three-Dimensional Self-Correcting Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the code constructors, C++20 thermal Monte Carlo and Forward Flux Sampling (FFS) simulation engines, MIT SubMIT Slurm orchestration scripts, raw benchmark logs (`V1`–`V7`), figure-rendering scripts, and compiled APS REVTeX 4.2 manuscripts investigating three-dimensional passive self-correcting quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Directory Layout

* **`paper/main.tex` & `paper/main.pdf`** — 6-page two-column APS REVTeX 4.2 manuscript (*Simulating Three-Dimensional Self-Correcting Quantum Memories: Minimal Contractive Cells, Multi-Scale Thresholds, and Thermal Cooling*).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — 2-page APS REVTeX 4.2 summary / proposal.
* **`cluster_v8/` & `cluster_v7/`** — Verified C++20 simulation suites and CSS code builders:
  * `kmc_engine_v8.cpp` / `kmc_engine_v7.cpp` — Rejection-free continuous-time Bortz–Kalos–Lebowitz (BKL) Glauber Kinetic Monte Carlo, 6-replica Forward Flux Sampling with exact ground-codeword committor relaxation, Union-Find cluster decoding, and zero-measurement storage self-healing / hot-state cooling benchmarks up to `56,312` qubits.
  * `build_v8_rigorous.py` & `build_v7_rescue.py` — Exact CSS welded code builders covering 5-scale single-sector and two-cycle quantum `2x2` cells (both provably contractive uniform `2x2` and thin-bridge `2x2`), pruned 3-arm T-bridge buttonhole tests, cubic-grid wire-stretching embeddings, `2 -> 1` sector pre-balancing, Haah's 3D Cubic Code, 3D Toric Code, and published Gu–Caha Random Layer Codes.
* **`src/`** — Exact finite-cell verifiers, result parsers, and clean SVG/PNG figure renderers:
  * `prove_exact_soundness.cpp` — Exhaustive single-pass proof over all `4,096` cell states establishing `c_Rep = 2/5, c_Par = 1` (`c_cycle = 2/5 < 1`) on uniform `2x2` four-arm cross cells (`exact_soundness_v8_report.txt`).
  * `render_v10_focused_figures.py` — Generates the 6 focused publication figures (`figures/fig_v10_1_classes.png` through `figures/fig_v10_6_cooling.png`).
  * `parse_v8_results.py` & `parse_v7_results.py` — Parse `cluster_v8_results.txt` and `cluster_v7_results.txt` into `parsed_v8_summary.txt` and `parsed_v7_summary.txt`.
  * `exact_rib_cleaning_theorem.py`, `test_sheaf_exact.py`, `test_classical_bdl.py`, `analyze_seeds_and_ribs.py` — Local cell-cleaning, CSS sheaf, and 2D classical dual-welded verifiers.
* **`cluster_pkg/`, `cluster_v2/`–`cluster_v6/`** — Earlier experimental iterations and benchmark logs (`cluster_*_results.txt`, `parsed_*_summary.txt`).
* **`figures/`** — Rendered SVG and high-DPI PNG figures (`fig_v9_1_classes.png`–`fig_v9_6_cooling.png` used in the final manuscripts, plus historical figure iterations).
* **`data/`** — Exported small/medium `.code` benchmark instances.

---

## Reproducing Figures and Manuscripts

### 1. Run Exact Cell-Contraction Proof, Parse Cluster Results & Render `V10` Figures
```bash
clang++ -O3 -std=c++20 src/prove_exact_soundness.cpp -o prove_exact_soundness && ./prove_exact_soundness
python3 src/parse_v8_results.py
python3 src/render_v10_focused_figures.py
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
