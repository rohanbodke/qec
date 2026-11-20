# Three-Dimensional Self-Correcting Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the recursive code constructors, case-free contraction and 3D Euclidean volume-packing verifiers, C++ continuous-time thermal Monte Carlo, Sparse-Markowitz Union-Find, fast linear-time local renormalization readout, and live syndrome zoom-out engines, MIT SubMIT Slurm scripts, raw benchmark logs (`V1`–`V12`), figure-rendering scripts (`V16`), and compiled APS REVTeX 4.2 manuscripts investigating three-dimensional passive self-correcting quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Key Files

* **`paper/main.tex` & `paper/main.pdf`** — Seven-page, five-figure APS REVTeX 4.2 manuscript (*Three-Dimensional Self-Correcting Quantum Memories: Minimal Welded Cells, Local Readout, and Euclidean Packing*).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — Two-page APS REVTeX 4.2 project summary / proposal.
* **`src/prove_analytic_contraction.py` & `src/prove_3d_packing_and_L3.py`** — Verifies the case-free `2 x 2` contraction proof (`W <= 2V`, `2/5` per cycle), the exact 3D Euclidean volume-packing bound (`2L^2 + 2L <= L^3 <=> L >= 1 + sqrt(3)`), `2/7` confining (`10/63` cycle) contraction on the minimal 3D-compatible `3 x 3` single-cross cell, and the single-step mixed `(X + Z)` logical shortcut no-go (`K = 1 -> 3 -> 49 -> 737`).
* **`src/render_v16_tasteful_figures.py`** — Generates the 5 publication figures (`figures/fig_v16_1_landscape.png` through `figures/fig_v16_5_cooling.png`).
* **`cluster_v9/kmc_engine_v10.cpp`** — C++ continuous-time thermal simulation engine with Sparse-Markowitz GF(2) Union-Find, linear-time `O(N)` local cell-by-cell readout (`rg_kmc`), alternating `(X, Z)` syndrome zoom-out (`RGV2`), and multi-scale gradual cooling (`cool`), verified up to `423,559` physical qubits.

---

## Reproducing Figures and Manuscripts

```bash
python3 src/prove_analytic_contraction.py
python3 src/prove_3d_packing_and_L3.py
python3 src/render_v16_tasteful_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
