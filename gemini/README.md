# Simulating Three-Dimensional Self-Correcting Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the code constructors, $O(N)$ recursive logical pushforward verifiers, C++ continuous-time thermal Monte Carlo, Forward Flux Sampling (FFS), and live six-level syndrome-renormalization engines, MIT SubMIT Slurm orchestration scripts, raw benchmark logs (`V1`–`V9`), figure-rendering scripts, and compiled APS REVTeX 4.2 manuscripts investigating three-dimensional passive self-correcting quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Directory Layout

* **`paper/main.tex` & `paper/main.pdf`** — 7-page two-column APS REVTeX 4.2 manuscript (*Simulating Three-Dimensional Self-Correcting Quantum Memories: Minimal Contractive Cells, Quarter-Million-Qubit Scaling, and Thermal Cooling*).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — 2-page APS REVTeX 4.2 summary / proposal.
* **`cluster_v9/`, `cluster_v8/` & `cluster_v7/`** — Verified C++ simulation suites and CSS code builders up to `379,976` physical qubits:
  * `kmc_engine_v9.cpp` / `kmc_engine_v8.cpp` — Rejection-free continuous-time Bortz–Kalos–Lebowitz (BKL) Glauber Kinetic Monte Carlo, live 6-level syndrome-only thermal RG defect-density coarse-graining (`s = 5 -> 0` on `322,159` planar and `288,344` torus qubits), 6-replica Forward Flux Sampling with exact ground-codeword committor relaxation, Union-Find cluster decoding, and zero-measurement storage self-healing / hot-state cooling benchmarks.
  * `build_v9_quarter_million.py` & `build_v8_rigorous.py` — Exact CSS welded code builders with $O(N)$ recursive logical operator propagation (`bad_comm = 0` verified up to `379,976` qubits), covering 6-level `2x2` ladders, 3-level full `(X, Z)` quantum ladders (`19 -> 1,942 -> 189,988` qubits), two-cycle uniform `2x2` (`47,387` qubits) and boundaryless torus (`50,296` qubits) codes, pruned 3-arm T-bridge buttonhole tests, cubic-grid wire-stretching embeddings, `2 -> 1` sector pre-balancing, Haah's 3D Cubic Code, 3D Toric Code, and published Gu–Caha Random Layer Codes.
* **`src/`** — Exact finite-cell verifiers, result parsers, and clean SVG/PNG figure renderers:
  * `prove_exact_soundness.cpp` — Exhaustive single-pass proof over all `4,096` cell states establishing `c_Rep = 2/5, c_Par = 1` (`c_cycle = 2/5 < 1`) on uniform `2x2` four-arm cross cells (`exact_soundness_v8_report.txt`).
  * `render_v11_final_figures.py` — Generates the 6 publication figures (`figures/fig_v11_1_classes.png` through `figures/fig_v11_6_cooling.png`).
  * `parse_v9_results.py` & `parse_v8_results.py` — Parse `cluster_v9_results.txt` and `cluster_v8_results.txt` into `parsed_v9_summary.txt` and `parsed_v8_summary.txt`.
* **`figures/`** — Rendered SVG and high-DPI PNG figures (`fig_v11_1_classes.png`–`fig_v11_6_cooling.png`).

---

## Reproducing Figures and Manuscripts

### 1. Run Exact Cell-Contraction Proof, Parse Cluster Results & Render `V11` Figures
```bash
clang++ -O3 -std=c++20 src/prove_exact_soundness.cpp -o prove_exact_soundness && ./prove_exact_soundness
python3 src/parse_v9_results.py
python3 src/render_v11_final_figures.py
```

### 2. Compile the APS REVTeX 4.2 Manuscripts
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
