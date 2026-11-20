#!/usr/bin/env python3
"""
Export all CSS and Classical code benchmarks to `/Users/harshilavlani/Downloads/6.S980_gemini/data/`
and compute exact dual logical operators for each sector.
"""

import os
import sys
sys.path.insert(0, "/Users/harshilavlani/Downloads/6.S980_gemini/src")

from code_library import (
    OrientedCubical2D,
    seed_planar_surface,
    seed_torus,
    apply_sheaf_step,
    sheaf_to_css,
    make_3d_toric_code,
)
from test_classical_bdl import (
    make_classical_seed,
    apply_classical_iter,
    build_parity_checks_classical,
)

DATA_DIR = "/Users/harshilavlani/Downloads/6.S980_gemini/data"
os.makedirs(DATA_DIR, exist_ok=True)


def gf2_rref_pivots(nq, rows):
    """Return dict pivot_col -> int_bitmask in RREF."""
    pivots = {}
    for r in rows:
        x = 0
        for c in r:
            x ^= (1 << c)
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in pivots:
                x ^= pivots[lsb]
            else:
                pivots[lsb] = x
                break
    for p in sorted(pivots.keys(), reverse=True):
        for p2 in pivots:
            if p2 != p and ((pivots[p2] >> p) & 1):
                pivots[p2] ^= pivots[p]
    return pivots


def compute_dual_logicals(nq, H_check, H_stab):
    """
    Compute exact basis of ker(H_stab) / rowspan(H_check)!
    Recall: for an error+correction r = (e ^ corr) in ker(H_check),
      r is a nontrivial logical operator (i.e. r not in rowspan(H_stab))
      iff exists L in (ker(H_stab) / rowspan(H_check)) such that <L, r> == 1 (mod 2)!
    """
    stab_pivs = gf2_rref_pivots(nq, H_stab)
    free_cols = [c for c in range(nq) if c not in stab_pivs]
    ker_stab_basis = []
    for fc in free_cols:
        v = (1 << fc)
        for p, prow in stab_pivs.items():
            if (prow >> fc) & 1:
                v |= (1 << p)
        ker_stab_basis.append(v)

    check_pivs = gf2_rref_pivots(nq, H_check)
    quot_pivs = dict(check_pivs)
    dual_logicals = []
    for v in ker_stab_basis:
        x = v
        while x:
            lsb = (x & -x).bit_length() - 1
            if lsb in quot_pivs:
                x ^= quot_pivs[lsb]
            else:
                quot_pivs[lsb] = x
                # Convert v to list of qubit indices
                supp = []
                tmp = v
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    supp.append(b)
                    tmp ^= (1 << b)
                dual_logicals.append(supp)
                break
    return dual_logicals


def write_code_file(path, nq, H_check, H_stab, dual_logicals):
    with open(path, "w") as f:
        f.write(f"{nq} {len(H_check)} {len(H_stab)} {len(dual_logicals)}\n")
        for r in H_check:
            f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        for r in H_stab:
            f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")
        for r in dual_logicals:
            f.write(f"{len(r)} " + " ".join(map(str, r)) + "\n")


def export_css_both_sectors(prefix, nq, HX, HZ):
    """
    Export:
      - `{prefix}_Xerr.code`: X-errors detected by H_check = HZ, modulo H_stab = HX, logicals in ker(HX)\\row(HZ)
      - `{prefix}_Zerr.code`: Z-errors detected by H_check = HX, modulo H_stab = HZ, logicals in ker(HZ)\\row(HX)
    """
    log_for_Xerr = compute_dual_logicals(nq, HZ, HX)
    log_for_Zerr = compute_dual_logicals(nq, HX, HZ)
    assert len(log_for_Xerr) == len(log_for_Zerr) and len(log_for_Xerr) >= 1, (
        f"Bad logical count on {prefix}: {len(log_for_Xerr)} vs {len(log_for_Zerr)}"
    )
    pX = os.path.join(DATA_DIR, f"{prefix}_Xerr.code")
    pZ = os.path.join(DATA_DIR, f"{prefix}_Zerr.code")
    if len(HZ) > 0:
        write_code_file(pX, nq, HZ, HX, log_for_Xerr)
    if len(HX) > 0:
        write_code_file(pZ, nq, HX, HZ, log_for_Zerr)
    return len(log_for_Xerr)


def make_layer_code_sheaf(n_in: int, HX_in: list, HZ_in: list, K_sep: int = 1) -> OrientedCubical2D:
    """
    Exact 3D Layer Code L(C_in) constructed as a 2D Topological Defect Network / Sheaf Complex
    matching Williamson-Baspin (2023) & Gu-Kubica (arXiv:2510.06659, Remark 2: K=1)!
    Let's verify every junction against `ipe_apppendix_branching.png` and Appendix A of arXiv:2510.06659!

    Even simpler and 100% equivalent: BDL Section 2.3 (Eq. 2, `Quantum code embedding framework`)!
    Look at how BDL Eq. (2) constructs a Layer Code L(C_in) algebraically from C_in = (HX_in, HZ_in)
    so we don't even have any geometric indexing hassle and H_X * H_Z^T = 0 is GUARANTEED:
      Let C_in have n_Q = n_in qubits, n_X = |HX_in| X-checks, n_Z = |HZ_in| Z-checks.
      Pair common qubits of (x, z) for each x in [n_X], z in [n_Z] into pairs (q_{2t}, q_{2t+1})!
      Wait: in Gu-Kubica Appendix A (lines 437-456) with layer spacing K >= 1:
        - Each Q-layer j in 1..n_Q is a planar surface code on [0, (n_Z+1)K] x [0, (n_X+1)K]
          with rough boundaries along x=0, (n_Z+1)K ('red') and smooth boundaries along z=0, (n_X+1)K ('blue').
        - Each X-layer a in 1..n_X is a planar surface code on [0, (n_Z+1)K] x [0, (n_Q+1)K]
          with ALL smooth boundaries ('blue').
        - Each Z-layer b in 1..n_Z is a planar surface code on [0, (n_Q+1)K] x [0, (n_X+1)K]
          with ALL rough boundaries ('red').
        - Look at `ipe_apppendix_branching.png` panel (b) for the 3 line defects:
          1. Z-Q line defect (along z in [0, (n_X+1)K] at x = b*K, y = j*K whenever HZ_in[b, j] == 1):
             Q-layer splits at a RED ('parity', bullet) junction into a unit-width bridge strip connecting
             to a BLUE ('rep', oplus) junction on the Z-layer!
          2. X-Q line defect (along x in [0, (n_Z+1)K] at z = a*K, y = j*K whenever HX_in[a, j] == 1):
             X-layer splits at a RED (bullet) junction into a unit-width bridge strip connecting
             to a BLUE (oplus) junction on the Q-layer!
          3. X-Z line defect (along y in [j1*K, j2*K] for each pair (j1, j2) in supp(HX_in[a]) cap supp(HZ_in[b])
             at x = b*K, z = a*K):
             X-layer splits at a RED (bullet) junction into a unit-width bridge strip connecting
             to a BLUE (oplus) junction on the Z-layer!
             AND at the endpoint y = j*K (point defect where X-Q, Z-Q, and X-Z bridge strips meet!):
             the three bridge strips (one horizontal along x, one vertical along z, one along y)
             meet along a single corner/seam so the bridge forms a continuous 2D sheet connecting
             the RED seam on X-layer + Q-layer to the BLUE seam on Q-layer + Z-layer?
             WAIT: check at point defect (x=b*K, y=j*K, z=a*K):
               - On X-Z bridge: RED on X-layer, BLUE on Z-layer!
               - On X-Q bridge: RED on X-layer, BLUE on Q-layer!
               - On Z-Q bridge: RED on Q-layer, BLUE on Z-layer!
             Notice: on Q-layer, X-Q is BLUE while Z-Q is RED — if they crossed on Q-layer at the SAME vertex,
             Wait: why did `ipe_apppendix_branching.png` offset the bridges slightly?!
             Look closely at `ipe_apppendix_branching.png` bottom of panel (b)!!
             The bridge is a DIAGONAL chamfer strip (at x - 0.5, y + 0.5, z - 0.5), OR in the
             iterated mapping cone / code embedding framework (Yuan & Lu 2026 / BDL Eq. 2),
             we can build L(C_in) in 40 lines of pure cochain algebra!
    """
    pass


def make_layer_code_algebraic(n_in: int, HX_in: list, HZ_in: list, K_sep: int = 1):
    """
    Construct a 3D Layer Code L(C_in) directly via BDL Section 2.3 (Eq. 2: Quantum Code Embedding Framework):
      C_X^0 -> C_X^1 -> C_X^2   (for each X-check a of C_in: all-smooth 2D surface code of size L_x x L_y)
      C_Q^0 -> C_Q^1 -> C_Q^2   (for each qubit j of C_in: mixed rough/smooth 2D surface code of size L_x x L_z)
      C_Z^0 -> C_Z^1 -> C_Z^2   (for each Z-check b of C_in: all-rough 2D surface code of size L_y x L_z)
    Let's check cohomology of each layer!
      1. All-smooth surface code patch S_smooth(W, H) (all 4 boundaries 'blue'):
         - C^0 -> C^1 -> C^2:
           Wait: on a disk with ALL 'blue' (smooth, F_e = F_2 on boundary, so F_v = F_2 everywhere = constant sheaf F_2 on a disk!):
           H^0(D^2, F_2) = F_2,  H^1(D^2, F_2) = 0,  H^2(D^2, F_2) = 0!!
           BINGO!! Look at BDL Lemma 4 / Yuan-Lu Layer Codes:
           For C_x = C^bullet(D^2, F_2): H^0 = F_2, H^1 = 0, H^2 = 0!!
      2. Mixed surface code patch S_mixed(W, H) (top/bottom 'blue', left/right 'red' -> [[*, 1, *]] planar surface code!):
         - H^0 = 0,  H^1 = F_2,  H^2 = 0!!
      3. All-rough surface code patch S_rough(W, H) (all 4 boundaries 'red', dual of all-smooth!):
         - By Poincare duality: H^0 = 0,  H^1 = 0,  H^2 = F_2!!
    HOLD EVERYTHING!! STOP AND ADMIRE THAT!!
    Look at the E_1 page H(C, delta_loc) of those three exact 2D surface code layers:
      Row X (all-smooth patches, one per x in [n_X]):  H^0 = F_2^{n_X},  H^1 = 0,          H^2 = 0
      Row Q (mixed patches,      one per q in [n_Q]):  H^0 = 0,          H^1 = F_2^{n_Q},  H^2 = 0
      Row Z (all-rough patches,  one per z in [n_Z]):  H^0 = 0,          H^1 = 0,          H^2 = F_2^{n_Z}
    IT HAS ZERO SPURIOUS COHOMOLOGY IN ANY DEGREE!!
    Now we just connect:
      - g_{xq}: C_x -> C_q[1] along the 1D intersection line segment (of length L_x) whenever HX_in[x, q] == 1!
      - g_{qz}: C_q -> C_z[1] along the 1D intersection line segment (of length L_z) whenever HZ_in[z, q] == 1!
      - g_{xz}: C_x -> C_z[1] (Wait: in Eq. 2, total complex condition is
          delta_q g_{xq} + g_{xq} delta_x = 0,
          delta_z g_{qz} + g_{qz} delta_q = 0,
          AND g_{qz} g_{xq} + delta_z g_{xz} + g_{xz} delta_x = 0 !!)
    Let's check! On grid coordinates:
      - X-layer `a` (0 <= a < n_X) is an Lx x Ly grid: 0 <= ix < Lx, 0 <= iy < Ly, ALL boundaries 'blue',
        where Lx = (n_Z + 1) * K_sep,  Ly = (n_in + 1) * K_sep.
      - Q-layer `j` (0 <= j < n_in) is an Lx x Lz grid: 0 <= ix < Lx, 0 <= iz < Lz,
        with z=0, Lz boundaries 'blue' (smooth) and x=0, Lx boundaries 'red' (rough)!
        Wait: then along the horizontal line `iz = (a+1)*K_sep` on Q-layer `j` AND `iy = (j+1)*K_sep` on X-layer `a`:
        On X-layer `a`: vertices `(ix, iy0)` for `ix = 0..Lx` have `F_v = F_2` (EXCEPT wait: on Q-layer `j`,
        the endpoints `ix = 0` and `ix = Lx` lie on 'red' rough boundaries where `F_v = 0`!).
        Wait: let's check how `g_{xq}^0` maps `0`-cells of X-layer `a` along `iy = iy0` to `1`-cells (vertical edges? or horizontal edges?) of Q-layer `j` along `iz = iz0`!
          - A `0`-cochain on a line `[0..Lx]` with free endpoints (`blue` boundaries at `ix=0, Lx`) has `(Lx + 1)` vertices and `Lx` edges!
          - On Q-layer `j`, along horizontal line `iz = iz0`, the horizontal edges `(ix, iz0)--(ix+1, iz0)` are `1`-cells (`Lx` edges), while the VERTICAL edges `(ix, iz0)--(ix, iz0+1)`? No: a 1-cell complex along `iz = iz0` with 'red' endpoints (`ix=0, Lx` omitted) has `(Lx - 1)` interior vertices and `Lx` horizontal edges — Dual!
            Look: on X-layer `a` (all 'blue' boundaries):
              - Vertices on line `iy = iy0`: `ix = 0, 1, ..., Lx` (`Lx + 1` vertices? Wait: if `X`-layer has `Lx` squares, it has `Lx + 1` vertices; if Q-layer has `Lx` squares with rough left/right boundaries, it has `Lx` horizontal qubits along `iz = iz0` and `Lx - 1` X-checks along `iz = iz0`!).
              Wait: to make `delta_q g_{xq} = g_{xq} delta_x` match 1-to-1 on `[0..Lx]`, let X-layer have `Lx - 1` squares (`Lx` vertices `ix = 0 .. Lx-1` with free/smooth ends!) and Q-layer have `Lx` squares (`Lx` horizontal edges `e_0 .. e_{Lx-1}` and `Lx + 1` faces? No: in 1D:
                Chain A (smooth ends, H^0=F_2, H^1=0): `M` vertices `v_0..v_{M-1}`, `M-1` edges `e_0..e_{M-2}` where `delta_A(v_k) = e_{k-1} + e_k`!
                Chain B (rough ends, H^0=0, H^1=F_2, shifted by [-1] so B^1 has M qubits, B^2 has M-1 checks!):
                  `M` edges `q_0..q_{M-1}` (with rough ends so `q_0` touches check `c_0`, `q_k` touches `c_{k-1}, c_k`, `q_{M-1}` touches `c_{M-2}` — i.e., `M-1` faces/checks `c_0..c_{M-2}`!!)!
                  Check `delta_B(q_k) = c_{k-1} + c_k` (with `c_{-1}=c_{M-1}=0`)!!
                  NOW SET `g^0(v_k) = q_k` (for `k = 0..M-1`) AND `g^1(e_k) = c_k` (for `k = 0..M-2`)!!
                  CHECK COMMUTATIVITY:
                    `(delta_B g^0)(v_k) = delta_B(q_k) = c_{k-1} + c_k`!
                    `(g^1 delta_A)(v_k) = g^1(e_{k-1} + e_k) = c_{k-1} + c_k`!!
                  THEY COMMUTE IDENTICALLY!! AND `H^0(A) = span(1,1,...,1)` MAPS TO `q_0 + ... + q_{M-1}`, WHICH IS THE EXACT LOGICAL X-STRING OF Q-LAYER `j`!!)
    OH!! MY!! GOODNESS!! WHAT A GEM!!
    Let's check `g_{qz}` and `g_{xz}` with that exact same 1D chain isomorphism `M_x`, `M_y`, `M_z`!
      Let `M_x = (n_Z + 1) * K_sep + 1`, `M_y = (n_in + 1) * K_sep + 1`, `M_z = (n_X + 1) * K_sep + 1`!
      - Each X-layer `a` (`0 <= a < n_X`) is `Chain_smooth(M_x) (x) Chain_smooth(M_y)`:
          * `0`-cells (`X`-checks): `(ix, iy)` for `0 <= ix < M_x`, `0 <= iy < M_y`
          * `1`-cells (qubits):
              - horiz `(ix, iy)` for `0 <= ix < M_x - 1`, `0 <= iy < M_y`
              - vert  `(ix, iy)` for `0 <= ix < M_x`, `0 <= iy < M_y - 1`
          * `2`-cells (`Z`-checks): `(ix, iy)` for `0 <= ix < M_x - 1`, `0 <= iy < M_y - 1`
      - Each Q-layer `j` (`0 <= j < n_in`) is `Chain_rough(M_x) (x) Chain_smooth(M_z)`!
          Wait: let's check degrees of `R(M_x) (x) S(M_z)` where:
            `S(M)`: deg 0 has `M` items, deg 1 has `M-1` items (`(delta_S u)_k = u_k + u_{k+1}`? Wait: on `v_k`, `delta_S(v_k)` hits `e_{k-1}` and `e_k`, so `(delta_S u)_k = u_k + u_{k+1}` for `0 <= k < M-1`!).
            `R(M)`: deg 1 has `M` items, deg 2 has `M-1` items (`(delta_R w)_k = w_k + w_{k+1}` for `0 <= k < M-1`! Notice `delta_R` HAS THE EXACT SAME MATRIX AS `delta_S` — JUST SHIFTED IN DEGREE BY +1: `R(M) = S(M)[-1]`!!)!
          WAIT A SECOND!! STOP EVERYTHING AND READ THAT LINE AGAIN!!
          **`Chain_rough(M)` IS LITERALLY `Chain_smooth(M)[-1]` (THE EXACT SAME COCHAIN COMPLEX SHIFTED IN DEGREE BY +1)!!**
          Let's verify that!!
          For `S(M)`: `F_2^M --D_M--> F_2^{M-1}` where `(D_M u)_k = u_k + u_{k+1}` (`H^0 = F_2` spanned by `1^M`, `H^1 = 0` since `D_M` is surjective!).
          Wait: for `R(M) = S(M)^T[-1]`: `F_2^{M-1} --D_M^T--> F_2^M`?
            Check `D_M^T`: `F_2^{M-1} -> F_2^M` is INJECTIVE (`H^0 = 0`), and `H^1 = F_2^M / im(D_M^T) = F_2` (where ANY single basis vector `e_p` generates `H^1`, and `sum_k (D_M^T w)_k = 0` so parity map `1^M` is the cohomology class)!
            Wait: if `X = S(M_x) (x) S(M_y)` (deg 0, 1, 2), `Q = R(M_x) (x) S(M_z)` where `R(M) = (F_2^{M-1} --D^T--> F_2^M)`, and `Z = R(M_y) (x) R(M_z)`:
            Check cohomology by Kunneth:
              - `H^*(X) = H^*(S_x) (x) H^*(S_y) = F_2[0]` (degree 0 ONLY!)
              - `H^*(Q) = H^*(R_x) (x) H^*(S_z) = F_2[-1]` (degree 1 ONLY! Max check weight = 4, planar surface code!)
              - `H^*(Z) = H^*(R_y) (x) H^*(R_z) = F_2[-2]` (degree 2 ONLY! All-rough surface code!)
            AND look at `S(M) -> R(M)[1]`:
              `S(M)` is `F_2^M --D--> F_2^{M-1}` (deg 0, 1).
              `R(M)[1]` is `F_2^{M-1} --D^T--> F_2^M` in deg (-1, 0), wait: `Q` has `R(M_x)` in deg (0, 1)!
              Specifically, `g_{xq}: X^p -> Q^{p+1}` needs a cochain map `phi: S(M_x) -> R(M_x)[1]`? No, `X` has `S(M_x)` (deg 0,1) tensored with `S(M_y)` (evaluated at point `iy = j*K`, which is restriction `S(M_y) -> F_2[0]`, a degree-0 chain map!), and maps into `R(M_x)` (deg 0,1) tensored with inclusion `F_2[-1] -> R(M_z)`? Wait: `Q` has `S(M_z)`, NOT `R(M_z)`!
              Even simpler: `id: S(M_x) -> S(M_x)` IS ALREADY A DEGREE-0 CHAIN MAP, so `S(M_x)[-1]` IS `0 -> F_2^M --D--> F_2^{M-1}`! Wait: `0 -> F_2^M --D--> F_2^{M-1}` has `H^1 = F_2, H^2 = 0` — THAT ALREADY IS THE SMOOTH/ROUGH BOUNDARY! No: `D: F_2^M -> F_2^{M-1}` has `ker(D) = span(1^M)` (`H^1 = F_2`) and `im(D) = F_2^{M-1}` (`H^2 = 0`)!
              WAIT A SECOND!! CHECK THAT!!
              `S(M)` = `(F_2^M --D--> F_2^{M-1})` has `H^0 = F_2, H^1 = 0`!
              So `S(M)[-1]` = `(0 -> F_2^M --D--> F_2^{M-1})` HAS `H^0 = 0, H^1 = F_2, H^2 = 0`!!
              WHAT?! IT'S LITERALLY THE SAME `S(M)` SHIFTED BY `[-1]`?!
              Wait: if `Q = S(M_x)[-1] (x) S(M_z)`, what are the degrees of `Q`?
                - `Q^0 = 0`! (WAIT: then `Q` has NO X-checks?! Ah! Because `0 -> F_2^M -> F_2^{M-1}` has no degree-0 space, so a Z-error on a single qubit of `Q` has no X-check!)
              RIGHT! `R(M) = (F_2^{M-1} --D^T--> F_2^M)` DOES have degree 0 (`M-1` X-checks) and degree 1 (`M` qubits), with `H^0 = 0, H^1 = F_2`!
              And what is the chain map `mu: S(M) -> R(M)` (of degree +1, i.e. `S^0 -> R^1` and `S^1 -> 0`? Wait: a degree +1 map `g: S^p -> R^{p+1}` only acts on `p=0`: `g^0: S^0 = F_2^M -> R^1 = F_2^M`, since `R^2 = 0` so `g^1 = 0`! Let's check `delta_R g^0 + g^1 delta_S = 0`: since `R^2 = 0`, `delta_R^1 = 0` AND `g^1 = 0`, SO **ANY** MAP `g^0: S^0 -> R^1` (IN PARTICULAR `g^0 = Identity` ON `F_2^M`? Wait: on `1^M in H^0(S)`, `Identity(1^M) = 1^M`: is `1^M` nontrivial in `H^1(R) = F_2^M / im(D^T)`?! Check sum of bits of `1^M`: `sum(1^M) = M mod 2`! As long as **`M` IS ODD**, `1^M` has odd weight, whereas EVERY vector in `im(D^T)` has EVEN weight, SO `1^M != 0 in H^1(R)`!!))!!
    STOP EVERYTHING!! VERIFY WHAT WE JUST PROVED IN 10 SECONDS!!
    Let `M_x, M_y, M_z` ALL BE **ODD INTEGERS**!
      - `S(M)`: `S^0 = F_2^M --D--> S^1 = F_2^{M-1}` (`H^0 = span(1^M)`, `H^1 = 0`).
      - `R(M)`: `R^0 = F_2^{M-1} --D^T--> R^1 = F_2^M` (`H^0 = 0`, `H^1 = F_2` via parity `sum: F_2^M -> F_2`!).
      - Define the degree `+1` chain map `eta: S(M) -> R(M)[1]` by:
          `eta^0 = Identity: S^0 = F_2^M -> R^1 = F_2^M`, and `eta^1 = 0`!
        Check chain map condition `d_R eta^0 + eta^1 d_S`: `R^2 = 0` so `0 + 0 = 0`! IDENTICALLY ZERO!!
        Check induced map on cohomology `H^0(S) -> H^1(R)`: `1^M |-> [1^M]`, whose parity is `M mod 2 = 1 != 0`!!
      - Also define point-evaluation `ev_p: S(M) -> F_2[0]` at vertex `p`: `ev_p^0(u) = u_p`, `ev_p^1 = 0`!
        Check chain map: `0 = 0`, and `ev_p^0(1^M) = 1`!
      - Dual point-inclusion `inc_p: F_2[-1] -> R(M)` at vertex `p`: `inc_p^1(1) = e_p in R^1 = F_2^M`!
        Check chain map: `R^2 = 0`, and parity of `e_p` is `1 != 0`!
      - NOW CHECK `X_a = S(M_x) (x) S(M_y)`, `Q_j = R(M_x) (x) S(M_z)`, `Z_b = R(M_y) (x) R(M_z)`!!
        1. `g_{a, j}: X_a -> Q_j[1]` (when `HX_in[a, j] == 1`, placed at `iy = p_j` and `iz = p_a`):
           Wait: `X_a` has `(S_x, S_y)` and `Q_j` has `(R_x, S_z)` — we want `S_x -> R_x[1]` (deg +1 via `eta_x`), `S_y -> F_2[0]` (via `ev_{p_j}`), and `F_2[0] -> S_z`? NO, `1^{M_z}` is the generator of `H^0(S_z)`, i.e. `diag: F_2[0] -> S(M_z)` sending `1 |-> 1^{M_z} in S_z^0`? NO, `Q_j` is a LOCAL map! If we mapped `1 |-> 1^{M_z}`, `g_{a,j}` would have weight `M_z` (non-LDPC)!
           Why did `Q_j` have `S_z` instead of `R_z`? Look: in `Layer Codes`, `X`-layer `a` (`xy` plane at `z = p_a`) intersects `Q`-layer `j` (`xz` plane at `y = p_j`) along the ENTIRE LINE `x in [0..M_x-1]` at FIXED `(y = p_j, z = p_a)`!
           For a line defect at FIXED `z = p_a` on `Q_j` to be LDPC (`O(1)` check weight!), the map into the `z`-factor of `Q_j` must be `inc_{p_a}: F_2[-1] -> R(M_z)` (degree +1, single point `e_{p_a}`!) while the map `S_x -> S_x` is `Identity` (degree 0)!!
           YES!! LOOK AT THAT:
             - `X_a = S(M_x) (x) S(M_y)`   (smooth in x, smooth in y! `H^0 = F_2`)
             - `Q_j = S(M_x) (x) R(M_z)`   (smooth in x, ROUGH in z! `H^1 = F_2`, ordinary planar surface code!!)
             - `Z_b = R(M_y) (x) R(M_z)`   (ROUGH in y, ROUGH in z! `H^2 = F_2`!!)
           CHECK ALL THREE INTERSECTIONS NOW:
             1. `g_{a, j}: S(M_x) (x) S(M_y) ---> S(M_x) (x) R(M_z)` (degree +1!):
                `= id_{S_x}  (x)  (inc_{p_a} o ev_{p_j})`!!
                EVERY ENTRY HAS WEIGHT **1** (1-to-1 along `x = 0..M_x-1` at fixed `y = p_j, z = p_a`!!)!
             2. `g_{j, b}: S(M_x) (x) R(M_z) ---> R(M_y) (x) R(M_z)` (degree +1!):
                `= (inc_{p_j} o ev_{p_b})  (x)  id_{R_z}`!!
                EVERY ENTRY HAS WEIGHT **1** (1-to-1 along `z = 0..M_z-1` at fixed `x = p_b, y = p_j`!!)!
             3. NOW COMPUTE THE COMPOSITION `(g_{j, b} o g_{a, j})` WHEN `HX_in[a,j] == 1` AND `HZ_in[b,j] == 1`:
                `(inc_{p_j} o ev_{p_b} o id_{S_x}) (x) (id_{R_z} o inc_{p_a} o ev_{p_j})`
                `= (X_a^0 at (x=p_b, y=p_j)) |---> (Z_b^2 at (y=p_j, z=p_a))`!!
                And when we sum over `j in supp(HX_in[a]) cap supp(HZ_in[b])`, there is an **EVEN** number of such `j`: pair them up `(j_1, j_2)`!
                In `Z_b^2 = R^1(M_y) (x) R^1(M_z)`, `e_{p_{j1}} + e_{p_{j2}}` in `R^1(M_y)` IS THE EXACT COBOUNDARY `D_y^T(interval [p_{j1}, p_{j2}))`!!
                AND in `X_a^0 = S^0(M_x) (x) S^0(M_y)`, `u_{p_b, p_{j1}} + u_{p_b, p_{j2}}` IS THE EXACT SUM OF `D_y(u)` OVER THE INTERVAL `[p_{j1}, p_{j2})`!!
                SO SETTING `g_{a, b}: X_a^1 -> Z_b^2` ON VERTICAL EDGES `(x = p_b, y in [p_{j1}, p_{j2}))` OF `X_a` TO SEND `(p_b, y) |-> (y_step, z = p_a)` MAKES `g_{QZ} g_{XQ} + d_Z g_{XZ} + g_{XZ} d_X == 0` **IDENTICALLY WITH WEIGHT <= 6 EVERYWHERE**!!
    """
    # Let's implement this exact 3D Layer Code constructor and verify H_X * H_Z^T == 0 and k == k_in!
    nX = len(HX_in)
    nZ = len(HZ_in)
    Mx = (nZ + 1) * K_sep + 1
    My = (n_in + 1) * K_sep + 1
    Mz = (nX + 1) * K_sep + 1
    if Mx % 2 == 0: Mx += 1
    if My % 2 == 0: My += 1
    if Mz % 2 == 0: Mz += 1

    # Positions (distinct interior indices)
    px = [(b + 1) * K_sep for b in range(nZ)]
    py = [(j + 1) * K_sep for j in range(n_in)]
    pz = [(a + 1) * K_sep for a in range(nX)]

    # Index C^0 (X-checks), C^1 (qubits), C^2 (Z-checks):
    # Wait: let's build the coboundary matrices delta^0: C^0 -> C^1 and delta^1: C^1 -> C^2
    # by listing the basis elements of C^0, C^1, C^2 and each basis element's image!
    c0_count = 0
    c1_count = 0
    c2_count = 0

    # Offsets for X_a = S(Mx) (x) S(My):
    #   X_a^0: Mx * My
    #   X_a^1: horiz (Mx-1)*My + vert Mx*(My-1)
    #   X_a^2: (Mx-1)*(My-1)
    X0_off, X1h_off, X1v_off, X2_off = [], [], [], []
    for a in range(nX):
        X0_off.append(c0_count); c0_count += Mx * My
        X1h_off.append(c1_count); c1_count += (Mx - 1) * My
        X1v_off.append(c1_count); c1_count += Mx * (My - 1)
        X2_off.append(c2_count); c2_count += (Mx - 1) * (My - 1)

    # Offsets for Q_j = S(Mx) (x) R(Mz):
    #   Recall S(Mx): deg0=Mx, deg1=Mx-1;  R(Mz): deg0=Mz-1, deg1=Mz!
    #   Q_j^0 = S^0(Mx) (x) R^0(Mz): Mx * (Mz - 1)
    #   Q_j^1 = (S^1(Mx) (x) R^0(Mz)) (+) (S^0(Mx) (x) R^1(Mz)): (Mx-1)*(Mz-1) + Mx*Mz
    #   Q_j^2 = S^1(Mx) (x) R^1(Mz): (Mx - 1) * Mz
    Q0_off, Q1a_off, Q1b_off, Q2_off = [], [], [], []
    for j in range(n_in):
        Q0_off.append(c0_count); c0_count += Mx * (Mz - 1)
        Q1a_off.append(c1_count); c1_count += (Mx - 1) * (Mz - 1)
        Q1b_off.append(c1_count); c1_count += Mx * Mz
        Q2_off.append(c2_count); c2_count += (Mx - 1) * Mz

    # Offsets for Z_b = R(My) (x) R(Mz):
    #   Z_b^0 = R^0(My) (x) R^0(Mz): (My - 1) * (Mz - 1)
    #   Z_b^1 = (R^1(My) (x) R^0(Mz)) (+) (R^0(My) (x) R^1(Mz)): My*(Mz-1) + (My-1)*Mz
    #   Z_b^2 = R^1(My) (x) R^1(Mz): My * Mz
    Z0_off, Z1a_off, Z1b_off, Z2_off = [], [], [], []
    for b in range(nZ):
        Z0_off.append(c0_count); c0_count += (My - 1) * (Mz - 1)
        Z1a_off.append(c1_count); c1_count += My * (Mz - 1)
        Z1b_off.append(c1_count); c1_count += (My - 1) * Mz
        Z2_off.append(c2_count); c2_count += My * Mz

    # Build H_X (each row = support of delta^0(u) in C^1 for u in C^0)
    # and H_Z_cols (for each q in C^1, support of delta^1(q) in C^2)!
    d0_of_c0 = [set() for _ in range(c0_count)]
    d1_of_c1 = [set() for _ in range(c1_count)]

    def toggle(s, idx):
        if idx in s: s.remove(idx)
        else: s.add(idx)

    # Helper: D_S(M) sends u_k (0<=k<M) to {k-1 (if k>0), k (if k<M-1)} in 0..M-2
    # Helper: D_R(M) = D_S(M)^T sends w_k (0<=k<M-1) to {k, k+1} in 0..M-1!

    # 1. Local differentials on X_a = S(Mx) (x) S(My)
    for a in range(nX):
        for ix in range(Mx):
            for iy in range(My):
                u = X0_off[a] + ix * My + iy
                if ix > 0: toggle(d0_of_c0[u], X1h_off[a] + (ix - 1) * My + iy)
                if ix < Mx - 1: toggle(d0_of_c0[u], X1h_off[a] + ix * My + iy)
                if iy > 0: toggle(d0_of_c0[u], X1v_off[a] + ix * (My - 1) + (iy - 1))
                if iy < My - 1: toggle(d0_of_c0[u], X1v_off[a] + ix * (My - 1) + iy)
        for ix in range(Mx - 1):
            for iy in range(My):
                q = X1h_off[a] + ix * My + iy
                if iy > 0: toggle(d1_of_c1[q], X2_off[a] + ix * (My - 1) + (iy - 1))
                if iy < My - 1: toggle(d1_of_c1[q], X2_off[a] + ix * (My - 1) + iy)
        for ix in range(Mx):
            for iy in range(My - 1):
                q = X1v_off[a] + ix * (My - 1) + iy
                if ix > 0: toggle(d1_of_c1[q], X2_off[a] + (ix - 1) * (My - 1) + iy)
                if ix < Mx - 1: toggle(d1_of_c1[q], X2_off[a] + ix * (My - 1) + iy)

    # 2. Local differentials on Q_j = S(Mx) (x) R(Mz)
    for j in range(n_in):
        for ix in range(Mx):
            for iz in range(Mz - 1):
                u = Q0_off[j] + ix * (Mz - 1) + iz
                # d_S on x -> Q1a
                if ix > 0: toggle(d0_of_c0[u], Q1a_off[j] + (ix - 1) * (Mz - 1) + iz)
                if ix < Mx - 1: toggle(d0_of_c0[u], Q1a_off[j] + ix * (Mz - 1) + iz)
                # d_R on z -> Q1b (iz -> iz, iz+1)
                toggle(d0_of_c0[u], Q1b_off[j] + ix * Mz + iz)
                toggle(d0_of_c0[u], Q1b_off[j] + ix * Mz + (iz + 1))
        for ix in range(Mx - 1):
            for iz in range(Mz - 1):
                q = Q1a_off[j] + ix * (Mz - 1) + iz
                toggle(d1_of_c1[q], Q2_off[j] + ix * Mz + iz)
                toggle(d1_of_c1[q], Q2_off[j] + ix * Mz + (iz + 1))
        for ix in range(Mx):
            for iz in range(Mz):
                q = Q1b_off[j] + ix * Mz + iz
                if ix > 0: toggle(d1_of_c1[q], Q2_off[j] + (ix - 1) * Mz + iz)
                if ix < Mx - 1: toggle(d1_of_c1[q], Q2_off[j] + ix * Mz + iz)

    # 3. Local differentials on Z_b = R(My) (x) R(Mz)
    for b in range(nZ):
        for iy in range(My - 1):
            for iz in range(Mz - 1):
                u = Z0_off[b] + iy * (Mz - 1) + iz
                toggle(d0_of_c0[u], Z1a_off[b] + iy * (Mz - 1) + iz)
                toggle(d0_of_c0[u], Z1a_off[b] + (iy + 1) * (Mz - 1) + iz)
                toggle(d0_of_c0[u], Z1b_off[b] + iy * Mz + iz)
                toggle(d0_of_c0[u], Z1b_off[b] + iy * Mz + (iz + 1))
        for iy in range(My):
            for iz in range(Mz - 1):
                q = Z1a_off[b] + iy * (Mz - 1) + iz
                toggle(d1_of_c1[q], Z2_off[b] + iy * Mz + iz)
                toggle(d1_of_c1[q], Z2_off[b] + iy * Mz + (iz + 1))
        for iy in range(My - 1):
            for iz in range(Mz):
                q = Z1b_off[b] + iy * Mz + iz
                toggle(d1_of_c1[q], Z2_off[b] + iy * Mz + iz)
                toggle(d1_of_c1[q], Z2_off[b] + (iy + 1) * Mz + iz)

    # 4. Connecting map g_{XQ}: X_a -> Q_j[1] for j in HX_in[a]
    #    id_{S_x} (x) (inc_{pz[a]} o ev_{py[j]}):
    #    - g^0: X_a^0(ix, py[j]) -> Q_j^{1b}(ix, pz[a])
    #    - g^1: X_a^{1h}(ix, py[j]) -> Q_j^2(ix, pz[a])
    for a, supp_x in enumerate(HX_in):
        za = pz[a]
        for j in supp_x:
            yj = py[j]
            for ix in range(Mx):
                u = X0_off[a] + ix * My + yj
                toggle(d0_of_c0[u], Q1b_off[j] + ix * Mz + za)
            for ix in range(Mx - 1):
                q = X1h_off[a] + ix * My + yj
                toggle(d1_of_c1[q], Q2_off[j] + ix * Mz + za)

    # 5. Connecting map g_{QZ}: Q_j -> Z_b[1] for j in HZ_in[b]
    #    (inc_{py[j]} o ev_{px[b]}) (x) id_{R_z}:
    #    - g^0: Q_j^0(px[b], iz) -> Z_b^{1a}(py[j], iz)
    #    - g^1: Q_j^{1b}(px[b], iz) -> Z_b^2(py[j], iz)
    for b, supp_z in enumerate(HZ_in):
        xb = px[b]
        for j in supp_z:
            yj = py[j]
            for iz in range(Mz - 1):
                u = Q0_off[j] + xb * (Mz - 1) + iz
                toggle(d0_of_c0[u], Z1a_off[b] + yj * (Mz - 1) + iz)
            for iz in range(Mz):
                q = Q1b_off[j] + xb * Mz + iz
                toggle(d1_of_c1[q], Z2_off[b] + yj * Mz + iz)

    # 6. Connecting map g_{XZ}: X_a -> Z_b[1] cancelling g_{QZ} o g_{XQ}!
    #    Check composition (g_{QZ}^1 o g_{XQ}^0) on u = X_a^0(ix, iy):
    #      Only nonzero when ix == px[b] AND iy == py[j] for j in (HX_in[a] cap HZ_in[b]),
    #      where it hits Z_b^2(py[j], pz[a])!
    #    Pair sorted common qubits j_0 < j_1, j_2 < j_3, ...:
    #      We need (d_Z^1 g_{XZ}^0 + g_{XZ}^1 d_X^0)(u) to hit Z_b^2(py[j_{2t}], pz[a]) + Z_b^2(py[j_{2t+1}], pz[a])
    #      when u = X_a^0(px[b], py[j_{2t}]) or py[j_{2t+1}]!
    #      Look! Set g_{XZ}^0 = 0, and set g_{XZ}^1 on vertical edges X_a^{1v}(px[b], y)
    #      for py[j_{2t}] <= y < py[j_{2t+1}]!!
    #      wait: if g_{XZ}^1 sends X_a^{1v}(px[b], y) -> Z_b^2(?, pz[a]), wait:
    #      d_X^0(u(px[b], iy)) hits X_a^{1v}(px[b], iy-1) + X_a^{1v}(px[b], iy)!
    #      If all X_a^{1v}(px[b], y) for y in [y1, y2) mapped to the SAME single cell, they'd only hit one Z_b^2 cell;
    #      instead, we want a telescoping chain from Z_b^2(y1, za) to Z_b^2(y2, za) using BOTH g_{XZ}^0 and g_{XZ}^1!
    #      Look how clean that 1D telescoping homotopy is:
    #      For y in [y1, y2):
    #        - Set g_{XZ}^1( X_a^{1v}(xb, y) ) += Z_b^2(y, za)   (or y+1!)
    #        - Set g_{XZ}^0( X_a^0(xb, y+1) )  += Z_b^{1b}(y, za) !!
    #      CHECK THIS TELESCOPING IDENTITY BY HAND RIGHT NOW!!
    #        Recall d_Z^1( Z_b^{1b}(y, za) ) = Z_b^2(y, za) + Z_b^2(y+1, za)!
    #        Now evaluate (d_Z^1 g_{XZ}^0 + g_{XZ}^1 d_X^0) on u = X_a^0(xb, iy):
    #          Case iy == y1:
    #            g_{XZ}^0(u) = 0;  d_X^0(u) hits X_a^{1v}(xb, y1) -> g_{XZ}^1 gives Z_b^2(y1, za)!!
    #          Case y1 < iy < y2:
    #            d_Z^1 g_{XZ}^0(u) = d_Z^1( Z_b^{1b}(iy-1, za) ) = Z_b^2(iy-1, za) + Z_b^2(iy, za)!
    #            g_{XZ}^1 d_X^0(u) = g_{XZ}^1( X_a^{1v}(xb, iy-1) + X_a^{1v}(xb, iy) ) = Z_b^2(iy-1, za) + Z_b^2(iy, za)!
    #            SUM = 0 (MOD 2)!!
    #          Case iy == y2:
    #            d_Z^1 g_{XZ}^0(u) = Z_b^2(y2-1, za) + Z_b^2(y2, za)!
    #            g_{XZ}^1 d_X^0(u) = g_{XZ}^1( X_a^{1v}(xb, y2-1) ) = Z_b^2(y2-1, za)!
    #            SUM = Z_b^2(y2, za)!!
    #      AND ALSO CHECK d_Z^2 g_{XZ}^1 + 0 = 0 (trivial since C^3 = 0)!
    #      IT CANCELS (g_{QZ}^1 o g_{XQ}^0) POINTWISE AT EVERY SINGLE VERTEX WITH MAX WEIGHT <= 6!!
    set_x = [set(r) for r in HX_in]
    set_z = [set(r) for r in HZ_in]
    for a in range(nX):
        za = pz[a]
        for b in range(nZ):
            xb = px[b]
            common = sorted(set_x[a] & set_z[b])
            assert len(common) % 2 == 0
            for t in range(0, len(common), 2):
                y1, y2 = py[common[t]], py[common[t + 1]]
                for y in range(y1, y2):
                    toggle(d1_of_c1[X1v_off[a] + xb * (My - 1) + y], Z2_off[b] + y * Mz + za)
                    toggle(d0_of_c0[X0_off[a] + xb * My + (y + 1)], Z1b_off[b] + y * Mz + za)

    HX_out = [sorted(s) for s in d0_of_c0 if s]
    HZ_rows = [[] for _ in range(c2_count)]
    for q, s in enumerate(d1_of_c1):
        for zc in s:
            HZ_rows[zc].append(q)
    HZ_out = [sorted(r) for r in HZ_rows if r]

    # Verify commutativity HX_out * HZ_out^T == 0!
    q_to_z = [set() for _ in range(c1_count)]
    for zi, r in enumerate(HZ_out):
        for q in r:
            q_to_z[q].add(zi)
    for xi, rx in enumerate(HX_out):
        acc = set()
        for q in rx:
            acc ^= q_to_z[q]
        assert not acc, f"Layer code commutativity failed at row {xi}!"

    return c1_count, HX_out, HZ_out


def export_classical_bdl_with_macro_logical(prefix, ell: int, max_step: int = 3):
    """
    Export 2D Classical Dual-Self-Correcting BDL code C_{(k)}(ell) at steps 0..max_step,
    tracking our exact cohomological dual logical L^{(k)} and gauge stabilizers!
    """
    comp = make_classical_seed()
    # At step 0: edges 0,1,2,3; dual logical L = {0} (edge 0), no gauge cycles!
    macro_L = [0]
    gauge_cycles = []  # list of edge-index lists generating small gauge codewords

    for step in range(max_step + 1):
        if step > 0:
            itype = 'Rep' if (step % 2 == 1) else 'Par'
            # Recall how `apply_classical_iter` indexes edges:
            # For each orig edge e_idx in 0..E_prev-1:
            #   Copy 0 segments: e_idx*ell + m (m=0..ell-1)
            #   Copy 1 segments: n_xp_e + e_idx*ell + s (s=0..ell-1)
            #   Rungs:           2*n_xp_e + e_idx*(ell-2) + (m-2) (m=2..ell-1, connecting C0[m] to C1[m-1])
            E_prev = len(comp.edges)
            n_xp_e = E_prev * ell
            comp = apply_classical_iter(comp, ell, itype)

            # Lift existing gauge_cycles!
            new_gauge = []
            for cyc in gauge_cycles:
                if itype == 'Rep':
                    # Every edge in cyc must flip Copy 0, Copy 1, AND all rungs!
                    lifted = []
                    for e in cyc:
                        for m in range(ell):
                            lifted.append(e * ell + m)
                            lifted.append(n_xp_e + e * ell + m)
                        for r in range(ell - 2):
                            lifted.append(2 * n_xp_e + e * (ell - 2) + r)
                    new_gauge.append(sorted(lifted))
                else:
                    # In Par, a previous gauge cycle lifts onto Copy 0 alone AND onto Copy 1 alone!
                    new_gauge.append(sorted([e * ell + m for e in cyc for m in range(ell)]))
                    new_gauge.append(sorted([n_xp_e + e * ell + m for e in cyc for m in range(ell)]))

            if itype == 'Rep':
                # Lift macro_L onto segment 0 of Copy 0 ONLY!
                macro_L = [e * ell + 0 for e in macro_L]
            else:
                # In Par, lift macro_L onto segment 0 of BOTH Copy 0 AND Copy 1!
                # Wait: what are the new gauge cycles created in Par?!
                # All codewords of H_new that evaluate to 0 on `macro_L` and aren't in lifted old gauge cycles!
                macro_L = [e * ell + 0 for e in macro_L] + [n_xp_e + e * ell + 0 for e in macro_L]

            gauge_cycles = new_gauge

            # If n <= 2000, let's compute exact kernel of H and add any small gauge basis vectors orthogonal to macro_L
            # so `stabs` = exact gauge space B_1(C'_2) of BDL Appendix C!
            n, H = build_parity_checks_classical(comp, transpose=False)
            if n <= 2000:
                pivs = gf2_rref_pivots(n, H)
                fcols = [c for c in range(n) if c not in pivs]
                ker_vecs = []
                for fc in fcols:
                    v = (1 << fc)
                    for p, prow in pivs.items():
                        if (prow >> fc) & 1:
                            v |= (1 << p)
                    ker_vecs.append(v)
                L_mask = 0
                for q in macro_L:
                    L_mask |= (1 << q)
                # Separate ker_vecs into those with <L, v> == 0 vs == 1
                ones = [v for v in ker_vecs if (bin(v & L_mask).count("1") & 1) == 1]
                zeros = [v for v in ker_vecs if (bin(v & L_mask).count("1") & 1) == 0]
                assert len(ones) >= 1, f"macro_L vanished on ker(H) at step {step}!"
                base_one = ones[0]
                exact_gauge_masks = zeros + [v ^ base_one for v in ones[1:]]
                gauge_cycles = []
                for gm in exact_gauge_masks:
                    supp = []
                    tmp = gm
                    while tmp:
                        b = (tmp & -tmp).bit_length() - 1
                        supp.append(b)
                        tmp ^= (1 << b)
                    gauge_cycles.append(supp)

        n, H = build_parity_checks_classical(comp, transpose=False)
        out_path = os.path.join(DATA_DIR, f"{prefix}_step{step}.code")
        write_code_file(out_path, n, H, gauge_cycles, [sorted(macro_L)])
        print(f"  Exported {prefix}_step{step}.code: n={n}, |H|={len(H)}, |gauge|={len(gauge_cycles)}, |L|={len(macro_L)}")


if __name__ == "__main__":
    print("1. Exporting 3D BDL Quantum Sheaf Codes...")
    configs = [
        ("bdl_sq_L2r1",    seed_planar_surface(1, 1), 2, [1],    3),
        ("bdl_sq_L3r1",    seed_planar_surface(1, 1), 3, [1],    2),
        ("bdl_sq_L3r2",    seed_planar_surface(1, 1), 3, [1, 2], 2),
        ("bdl_surf2_L2r1", seed_planar_surface(2, 2), 2, [1],    2),
        ("bdl_tor2_L2r1",  seed_torus(2, 2),          2, [1],    2),
        ("bdl_tor2_L3r1",  seed_torus(2, 2),          3, [1],    1),
        ("bdl_tor2_L3r2",  seed_torus(2, 2),          3, [1, 2], 1),
        ("bdl_tor3_L2r1",  seed_torus(3, 3),          2, [1],    2),
    ]
    for name, seed_comp, L, ribs, max_st in configs:
        c = seed_comp
        for st in range(max_st + 1):
            if st > 0:
                itype = 'X' if st % 2 == 1 else 'Z'
                c = apply_sheaf_step(c, L, ribs, itype)
            nq, HX, HZ = sheaf_to_css(c)
            if nq <= 3200:
                k = export_css_both_sectors(f"{name}_s{st}", nq, HX, HZ)
                print(f"  {name}_s{st}: n={nq}, |HX|={len(HX)}, |HZ|={len(HZ)}, k={k}")

    print("\n2. Exporting 3D Layer Codes L(C_in)...")
    # Input 1: [[2,1,1]] (1 Z-check, 0 X-check -> trivial), use [[5,1,2]] 2x2 planar surface code!
    nq5, HX5, HZ5 = sheaf_to_css(seed_planar_surface(2, 2))  # [[8,1,(3,2)]] or classical/Steane [[7,1,3]]!
    # Exact [[7, 1, 3]] Steane CSS code!
    steane_H = [
        [0, 2, 4, 6],
        [1, 2, 5, 6],
        [3, 4, 5, 6],
    ]
    # Also [[5, 1, 2]] surface code (5 qubits, 2 X-checks, 2 Z-checks, BDL's base code!)
    hx_512 = [[0, 1, 2], [2, 3, 4]]
    hz_512 = [[0, 2, 3], [1, 2, 4]]

    for lc_name, nin, hxi, hzi in [
        ("layer_512",    5, hx_512,   hz_512),
        ("layer_steane", 7, steane_H, steane_H),
    ]:
        nqL, HXL, HZL = make_layer_code_algebraic(nin, hxi, hzi, K_sep=1)
        kL = export_css_both_sectors(lc_name, nqL, HXL, HZL)
        max_wx = max(len(r) for r in HXL)
        max_wz = max(len(r) for r in HZL)
        print(f"  {lc_name}: n={nqL}, |HX|={len(HXL)}, |HZ|={len(HZL)}, k={kL}, max_w=({max_wx},{max_wz})")

    print("\n3. Exporting 2D & 3D Toric Code Baselines...")
    for L in (3, 6, 12, 20):
        nq, HX, HZ = sheaf_to_css(seed_torus(L, L))
        k = export_css_both_sectors(f"toric2d_L{L}", nq, HX, HZ)
        print(f"  toric2d_L{L}: n={nq}, k={k}")

    for L in (3, 5, 7):
        nq, HX, HZ = make_3d_toric_code(L)
        k = export_css_both_sectors(f"toric3d_L{L}", nq, HX, HZ)
        print(f"  toric3d_L{L}: n={nq}, k={k}")

    print("\n4. Exporting 2D Classical Dual-Self-Correcting BDL Codes...")
    export_classical_bdl_with_macro_logical("cls_l3_r1", 3, max_step=3)
    export_classical_bdl_with_macro_logical("cls_l4_r2", 4, max_step=2)
