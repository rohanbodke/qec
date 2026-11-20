# Three-Dimensional Self-Correcting Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the recursive code constructors, case-free contraction verifiers, C++ continuous-time thermal Monte Carlo, Sparse-Markowitz Union-Find, fast linear-time local renormalization readout, and live six-level syndrome zoom-out engines, MIT SubMIT Slurm scripts, raw benchmark logs (`V1`–`V11`), explanatory figure-rendering scripts (`V14`), and compiled APS REVTeX 4.2 manuscripts investigating three-dimensional passive self-correcting quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Key Files

* **`paper/main.tex` & `paper/main.pdf`** — Two-column APS REVTeX 4.2 manuscript (*Three-Dimensional Self-Correcting Quantum Memories: Minimal Welded Cells, Fixed Thermal Thresholds, and Passive Cooling*).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — Two-page APS REVTeX 4.2 summary / proposal.
* **`src/prove_analytic_contraction.py` & `src/prove_exact_soundness.cpp`** — Verifies the short, zero-casework geometric proof (`W <= 2V` and sheet-comparison bridge cancellation) that a uniform `2 x 2` four-arm cross cell shrinks coarse syndromes by `2/5` per cycle (`174x` smaller per logical qubit than scale-8).
* **`src/render_v14_explanatory_figures.py` & `src/parse_v11_results.py`** — Generates the 7 large, visually self-explanatory publication figures (`figures/fig_v14_1_classes.png` through `figures/fig_v14_7_cooling.png`) and parses the `V11` campaign (`cluster_v11_results.txt`).
* **`cluster_v9/kmc_engine_v10.cpp` & `cluster_v9/build_v11_rg_and_cooling.py`** — C++ thermal simulation engine with exact Sparse-Markowitz GF(2) Union-Find, BDL's fast local renormalization readout (`rg_kmc`), alternating `(X, Z)` syndrome zoom-out (`RGV2`), and multi-scale gradual cooling (`cool`), verified up to `423,559` physical qubits.

---

## Reproducing Figures and Manuscripts

```bash
python3 src/prove_analytic_contraction.py
python3 src/parse_v11_results.py > parsed_v11_summary.txt
python3 src/render_v14_explanatory_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
