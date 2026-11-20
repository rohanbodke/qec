#!/usr/bin/env python3
"""
prove_analytic_contraction.py
Verifies each step of the short, case-free analytic proof that the uniform 2x2 cell
with a single central four-arm bridge cross contracts coarse-grained syndromes by
at most 2/5 per full (X, Z) welding cycle:

  1. Confining step (`W <= 2 V`, so wall-fraction `W / 5 <= (2/5) V`):
     - A doubled 2x2 cell has 8 sheet quadrants A_{c,i} (c in {0,1}, i in {0..3})
       and 4 bridge-arm faces B_d (d in {N, S, W, E}), with 5 boundary sub-edges per wall
       (2 on Sheet 0, 2 on Sheet 1, 1 on the bridge; 20 wall incidences total).
     - Both W = min(W_1, 20 - W_1) and V (the number of non-constant interior checks among
       the 8 sheet-bridge seams and 1 central 4-arm cross) are invariant under global bit
       inversion m -> 1 - m. Invert if needed so s_B = sum_d B_d <= 2.
     - If s_B = 0: on each sheet c, k active quadrants touch at least k of the 4 seams, so
       sum_i A_{c,i} <= V_c, giving W <= W_1 = 2(sum A_0 + sum A_1) <= 2(V_0 + V_1) = 2V.
     - If s_B = 2: the central cross is violated (v_cross = 1), and on each sheet c no
       quadrant assignment can agree with two 1-arms and two 0-arms without violating at
       least 2 seams (V_c >= 2). Thus V >= 2 + 2 + 1 = 5, so W <= 10 <= 2V automatically!
     - If s_B = 1: v_cross = 1 and V_c >= 1 for every A_c. Moreover sum(A_c) <= V_c for every
       A_c except A_c = (1,1,1,1) where sum(A_c) = 4 and V_c = 3 (and A_c = (0,0,0,0) has
       sum = 0, V_c = 1). Hence either V >= 5 (giving W <= 10 <= 2V), or at most one sheet is
       (1,1,1,1) while the other is (0,0,0,0) (giving W = 9 <= 10 = 2*5), or neither sheet is
       (1,1,1,1) (giving W <= W_1 <= 2(V_0 + V_1) + 1 = 2V - 1 < 2V).

  2. Dual step (`W_par <= 2 V_par`, so wall-fraction `W_par / 2 <= 1 * V_par`):
     - Define the sheet-difference bits d_i = A_{0,i} ^ A_{1,i} on the 4 quadrants.
     - Across any seam s separating quadrants i and j, XORing the two copies' 3-way parity
       checks cancels the bridge face identically:
         (A_{0,i} ^ A_{0,j} ^ B_s) ^ (A_{1,i} ^ A_{1,j} ^ B_s) = d_i ^ d_j.
       Hence whenever d_i != d_j across seam s, at least one of the two seam checks is violated!
     - Around the 4-seam square cycle, W_par = min(2*sum(d), 8 - 2*sum(d)) in {0, 2, 4}:
       whenever d is non-constant (W_par in {2, 4}), d changes value across at least 2 seams
       of the 4-cycle, forcing V_par >= 2 and therefore W_par <= 4 <= 2 V_par.

Combining both steps gives full-cycle syndrome contraction (2/5) * 1 = 2/5 < 1.
"""

def main():
    edges = [
        (0, 1, 8), (2, 3, 9), (0, 2, 10), (1, 3, 11),  # Sheet 0 seams
        (4, 5, 8), (6, 7, 9), (4, 6, 10), (5, 7, 11),  # Sheet 1 seams
        (8, 9, 10, 11),                                 # Central 4-arm bridge cross
    ]
    for m in range(1 << 12):
        b = [(m >> i) & 1 for i in range(12)]
        if sum(b[8:]) > 2:
            b = [1 - x for x in b]
        sB = sum(b[8:])
        assert sB in (0, 1, 2)
        V0 = sum(1 for e in edges[:4] if len({b[i] for i in e}) > 1)
        V1 = sum(1 for e in edges[4:8] if len({b[i] for i in e}) > 1)
        vc = 1 if len({b[i] for i in edges[8]}) > 1 else 0
        V = V0 + V1 + vc
        W1 = 2 * sum(b[:8]) + sB
        W = min(W1, 20 - W1)

        if sB == 0:
            assert sum(b[:4]) <= V0 and sum(b[4:8]) <= V1
        elif sB == 2:
            assert vc == 1 and V0 >= 2 and V1 >= 2 and V >= 5
        elif sB == 1:
            assert vc == 1 and V0 >= 1 and V1 >= 1
            for sheet in (b[:4], b[4:8]):
                vs = sum(1 for (x, y, z) in [
                    (sheet[0], sheet[1], b[8]), (sheet[2], sheet[3], b[9]),
                    (sheet[0], sheet[2], b[10]), (sheet[1], sheet[3], b[11])
                ] if len({x, y, z}) > 1)
                if sheet != [1, 1, 1, 1]:
                    assert sum(sheet) <= vs
                else:
                    assert sum(sheet) == 4 and vs == 3
        assert W <= 2 * V

        # Dual parity step check via XOR bridge cancellation:
        d = [b[i] ^ b[4 + i] for i in range(4)]
        Wp = min(2 * sum(d), 8 - 2 * sum(d))
        Vp = sum(1 for e in edges if (sum(b[i] for i in e) & 1))
        cyc_changes = (d[0] ^ d[1]) + (d[2] ^ d[3]) + (d[0] ^ d[2]) + (d[1] ^ d[3])
        if Wp > 0:
            assert cyc_changes >= 2 and Vp >= 2
        assert Wp <= 2 * Vp

    print("ALL ANALYTIC PROOF INEQUALITIES VERIFIED (c_Rep = 2/5, c_Par = 1, c_cycle = 2/5)!")


if __name__ == "__main__":
    main()
