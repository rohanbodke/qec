# Three-Dimensional Self-Correcting Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the recursive code constructors, case-free contraction and 3D volume-packing verifiers, C++ continuous-time thermal Monte Carlo, Sparse-Markowitz Union-Find, fast linear-time local renormalization readout, and live six-level syndrome zoom-out engines, MIT SubMIT Slurm scripts, raw benchmark logs (`V1`–`V12`), explanatory figure-rendering scripts (`V15`), and compiled APS REVTeX 4.2 manuscripts investigating three-dimensional passive self-correcting quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Key Files

* **`paper/main.tex` & `paper/main.pdf`** — Two-column APS REVTeX 4.2 manuscript (*Three-Dimensional Self-Correcting Quantum Memories: Minimal Welded Cells, Euclidean Packing, and High-Rate Cooling*).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — Two-page APS REVTeX 4.2 summary / proposal.
* **`src/prove_analytic_contraction.py` & `src/prove_3d_packing_and_L3.py`** — Verifies the case-free `2 x 2` contraction proof (`2/5` per cycle), the exact 3D Euclidean volume-packing bound (`2L^2 + 2L <= L^3 <=> L >= 1 + sqrt(3)`), exact `10/63` contraction on the minimal 3D-compatible `3 x 3` single-cross cell, and the mixed single-step `(X + Z)` logical proliferation no-go (`K = 1 -> 3 -> 49 -> 737`).
* **`src/render_v15_complete_figures.py` & `src/parse_v12_results.py`** — Generates the 8 large, visually self-explanatory publication figures (`figures/fig_v15_1_classes.png` through `figures/fig_v15_8_cooling.png`) and parses the `V12` campaign (`cluster_v12_results.txt`).
* **`cluster_v9/kmc_engine_v10.cpp`, `cluster_v9/build_v11_rg_and_cooling.py`, & `cluster_v9/build_v12_euclidean_and_highrate.py`** — C++ thermal simulation engine and multi-scale constructors verified up to `423,559` physical qubits, including bounded wire-energy hills on collision-free `Z^3` cubic grids (`56,312`q), compact sector-balanced cells (`586`q to `129,148`q), and `{4, 5}` / `{4, 6}` hyperbolic square-tiling welded memories (`K = 6` and `K = 10`).

---

## Reproducing Figures and Manuscripts

```bash
python3 src/prove_analytic_contraction.py
python3 src/prove_3d_packing_and_L3.py
python3 src/parse_v11_results.py > parsed_v11_summary.txt
python3 src/parse_v12_results.py > parsed_v12_summary.txt
python3 src/render_v15_complete_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
