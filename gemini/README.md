# Small Recursive Welding Cells: Local Cut Bounds, Hierarchical Readout, and Euclidean Wire Dilution (MIT 6.S980)

**Authors:** Harshil Avlani, Rohan Bodke, Rushil Shah, Yichen Li

This directory (`gemini/`) contains the cellular-sheaf CSS code constructors, standalone algebraic and thermodynamic verification certificates, C++ continuous-time heat-bath Monte Carlo and paired global/hierarchical readout engine (`kmc_engine_v13.cpp`), MIT SubMIT Slurm scripts, raw benchmark logs (`cluster_v13_results.txt`), uncertainty-aware parsers (`parse_v13_results.py`), publication figure scripts (`render_v17_referee_figures.py`), and compiled APS REVTeX 4.2 manuscripts investigating bounded-degree and three-dimensional Euclidean welded quantum memories ([Balasubramanian, Davydova, & Lin, arXiv:2605.10943](https://arxiv.org/abs/2605.10943); [Hayden, arXiv:2611.21749](https://arxiv.org/abs/2611.21749); [Gu et al., Phys. Rev. Lett. 136, 210601 (2026)](https://arxiv.org/abs/2510.06659)).

---

## Key Files

* **`paper/main.tex` & `paper/main.pdf`** — Nine-page, four-figure APS REVTeX 4.2 manuscript (*Small Recursive Welding Cells: Local Cut Bounds, Hierarchical Readout, and Euclidean Wire Dilution*), including the complete experiment and censoring manifest (`Table I`).
* **`proposal_improved.tex` & `proposal_improved.pdf`** — Two-page APS REVTeX 4.2 project summary aligned with `V17`.
* **`src/reproducibility_certificates.py` & `certificates_v17.json`** — Standalone 5-second verification suite certifying:
  1. Exact $2^{12}$ (`2x2`) and vectorized $2^{24}$ (`3x3`) single-cross auxiliary cut (`W <= 2 V_rep`, `min_{V>0} V_rep = 2`) and `(2L+1)`-bit wall-majority rounding (`R_mod <= 2/(2L+1) V_rep`) bounds with SHA-256 truth-table checksums.
  2. Physical CSS coboundary comparison (`V_rep <= |H_Z^T m|_int <= (11/5) V_rep`) on `R_Z` and exact periodic-torus meta-check pocket dimensions (`V_0 + 1` for 1-rib vs. `V_0 + E_0 + F_0 + 1` for 2-rib [BDL Sec. 6.2.6]).
  3. Exact MILP dressed subsystem distances for the pruned 3-arm T-bridge (`(6, 5)` vs. `(11, 10)`) and 1-step simultaneous blue+red sheaf cross (`d_Z^sub = 1` across all `K = 3` classes).
  4. Verified face/qubit and Tanner-diameter recurrences (`D_q <= 3 L^q`) underlying the conditional $\mathbb{R}^3$ interaction-range obstruction (`r_q = Omega(N_q^{0.0544})` at `L = 2`).
  5. Continuous-time generator detailed balance (`< 1e-18` on `[[8,2,2]]`), exact matrix-inverse mean first-passage times on `[[4,1,2]]` vs. C++ BKL, and exact finite-size check-wire Gibbs marginalization (`< 1.3e-16` TV residual) at $\beta_{\mathrm{eff}}(w) = 2\,\operatorname{arctanh}[\tanh^w(\beta/2)]$.
* **`cluster_v9/kmc_engine_v13.cpp` & `cluster_v9/build_v13_referee_upgrade.py`** — Rejection-free continuous-time heat-bath BKL engine with full failure-time list output (`t_list`), paired identical-seed global cluster-GF(2) vs. `HierarchicalRG` trajectory & i.i.d. single-shot benchmarks (`paired_kmc`, `iid_paired`), count-resolved coarse-syndrome flow (`rg_flow_v13`), and controlled check-wire subdivision codes (`w in {1, 2, 4, 8, 16}`).
* **`cluster_v13_results.txt`, `src/parse_v13_results.py`, & `parsed_v13_data.json`** — Complete `343/343` zero-error MIT SubMIT job `7419526` outputs, exact two-sector survival integrals $\hat{\tau}_Q = \int \hat{S}_X(t)\hat{S}_Z(t)\,dt$ with bootstrap SEMs and censoring audits, Clopper–Pearson 95% bounds, and weighted activation-slope fits.
* **`src/render_v17_referee_figures.py`** — Renders the four publication vector/raster figures (`figures/fig_v17_1_cell_and_decoder.{pdf,png}` through `figures/fig_v17_4_wire_dilution.{pdf,png}`).

---

## Reproducing Certificates, Figures, and Manuscripts

```bash
python3 src/reproducibility_certificates.py
python3 src/parse_v13_results.py
python3 src/render_v17_referee_figures.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd .. && cp paper/references.bib references.bib && pdflatex proposal_improved.tex && bibtex proposal_improved && pdflatex proposal_improved.tex && pdflatex proposal_improved.tex
```
