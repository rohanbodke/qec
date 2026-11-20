# Three-Dimensional Self-Correcting Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the code constructors, case-free contraction verifiers, C++ continuous-time thermal Monte Carlo, Forward Flux Sampling (FFS), and live six-level syndrome zoom-out engines, MIT SubMIT Slurm scripts, raw benchmark logs (`V1`–`V10`), figure-rendering scripts, and compiled APS REVTeX 4.2 manuscripts investigating three-dimensional passive self-correcting quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Key Files

* **`paper/main.tex` & `paper/main.pdf`** — Two-column APS REVTeX 4.2 manuscript (*Three-Dimensional Self-Correcting Quantum Memories: Minimal Welded Cells, Fixed Thermal Thresholds, and Passive Cooling*).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — Two-page APS REVTeX 4.2 summary / proposal.
* **`src/prove_analytic_contraction.py` & `src/prove_exact_soundness.cpp`** — Verifies the short, case-free analytic proof (`W <= 2V` and XOR bridge cancellation) that a uniform `2 x 2` four-arm cross cell contracts coarse syndromes by `2/5` per cycle (`174x` smaller per logical qubit than scale-8).
* **`src/render_v12_crystal_figures.py`** — Generates the 7 publication figures (`figures/fig_v12_1_classes.png` through `figures/fig_v12_7_cooling.png`).
* **`cluster_v9/` & `cluster_v8/`** — C++ thermal simulation engine (`kmc_engine_v9.cpp`) and linear-time recursive welded code builders (`build_v9_quarter_million.py`, `build_v10_unified.py`) verified up to `423,559` physical qubits (`cluster_v8_results.txt`, `cluster_v9_results.txt`, `cluster_v10u_results.txt`).

---

## Reproducing Figures and Manuscripts

```bash
python3 src/prove_analytic_contraction.py
python3 src/render_v12_crystal_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
