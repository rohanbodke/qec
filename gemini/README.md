# Small Cells and Wire Costs in Welded Quantum Memories (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the recursive welded CSS code constructors, standalone algebraic and thermodynamic verification certificates, C++ continuous-time heat-bath Monte Carlo and paired global/local readout engine (`kmc_engine_v13.cpp`), MIT SubMIT Slurm scripts, raw benchmark logs (`cluster_v13_results.txt`), figure scripts (`render_v18_clear_figures.py`), and compiled APS REVTeX 4.2 manuscripts investigating bounded-degree and three-dimensional Euclidean welded quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Key Files

* **`paper/main.tex` & `paper/main.pdf`** — Streamlined seven-page, five-figure APS REVTeX 4.2 manuscript (*Small Cells and Wire Costs in Welded Quantum Memories*), featuring concise prose, zero-casework wall contraction, paired readout validation, two-sector quantum lifetimes up to `423,559` qubits, the 3D packing bound, and the universal wire effective-bath collapse.
* **`proposal_improved.tex` & `proposal_improved.pdf`** — Two-page APS REVTeX 4.2 project summary.
* **`src/render_v18_clear_figures.py`** — Renders the five publication figures (`figures/fig_v18_1_welding.{pdf,png}` through `figures/fig_v18_5_wires.{pdf,png}`).
* **`src/reproducibility_certificates.py` & `certificates_v17.json`** — Standalone 5-second verification suite certifying single-cell wall contraction, physical coboundary & periodic-torus pocket invariants, MILP subsystem distances for the two tested simplifications, diameter recurrences, generator detailed balance, and exact check-wire Gibbs marginalization.
* **`cluster_v9/kmc_engine_v13.cpp`, `cluster_v13_results.txt`, & `src/parse_v13_results.py`** — Rejection-free continuous-time BKL engine, complete `343/343` zero-error SubMIT outputs, paired identical-seed global vs. local readout benchmarks, and wire-length sweeps (`w = 1, 2, 4, 8, 16`).

---

## Reproducing Certificates, Figures, and Manuscripts

```bash
python3 src/reproducibility_certificates.py
python3 src/parse_v13_results.py
python3 src/render_v18_clear_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && cp paper/references.bib references.bib && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
