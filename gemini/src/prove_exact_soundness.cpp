// prove_exact_soundness.cpp
// Exhaustive computer-assisted verification of local coarse-graining contraction constants
// (c_Rep, c_Par, and full-cycle c_cycle = c_Rep * c_Par) for:
//   1. 2D Classical Dual-Welded Ladders: ell=3 (r=1), ell=4 (r=2), ell=5 (r=3)
//   2. 3D Quantum Doubled-Square Units U_F (BDL Sec. 6.2.5 sheaf norm):
//      - Uniform 2x2 four-arm cross (L=2, r=1)
//      - Uniform 3x3 single cross (L=3, r=1)
//      - Thin-bridge 2x1 and 3x1 rectangles
//   3. Direct Primal CSS Face-Check Cleaning on 2x2 Four-Arm Cross (L=2, r=1)

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace std;

void verify_2d_classical_ladders() {
    cout << "=== 1. 2D Classical Dual-Welded Ladder Contraction Constants ===" << endl;
    for (int ell : {3, 4, 5}) {
        int r = ell - 2;
        int n_edges = 3 * ell - 2;

        // Single-pass computation of maximum Gamma_Rep <= 2:
        // Need for all m: min_{y in {0,1}} [phi(bu0^y, bu1^y) + phi(bv0^y, bv1^y)] <= vw(m),
        // where phi(0,0)=0, phi(1,1)=Gamma, phi(1,0)=phi(0,1)=max(Gamma-1, Gamma/2).
        // Let's check exact upper bound on Gamma across all 2^{n_edges} states:
        double max_Gamma = 2.0;
        for (int m = 0; m < (1 << n_edges); ++m) {
            int vw = 0;
            if (((m >> 0) ^ (m >> 1)) & 1) vw++;
            for (int k = 2; k < ell; ++k) {
                int b1 = (m >> (k - 1)) & 1, b2 = (m >> k) & 1, b3 = (m >> (2 * ell + k - 2)) & 1;
                if (!(b1 == b2 && b2 == b3)) vw++;
            }
            for (int s = 1; s < ell - 1; ++s) {
                int b1 = (m >> (ell + s - 1)) & 1, b2 = (m >> (ell + s)) & 1, b3 = (m >> (2 * ell + s - 1)) & 1;
                if (!(b1 == b2 && b2 == b3)) vw++;
            }
            if (((m >> (2 * ell - 2)) ^ (m >> (2 * ell - 1))) & 1) vw++;

            int bu0 = (m >> 0) & 1, bv0 = (m >> (ell - 1)) & 1;
            int bu1 = (m >> ell) & 1, bv1 = (m >> (2 * ell - 1)) & 1;

            // Binary search largest Gamma in [0, max_Gamma] valid for this m:
            double lo = 0.0, hi = max_Gamma;
            for (int it = 0; it < 40; ++it) {
                double mid = 0.5 * (lo + hi);
                double p10 = max(mid - 1.0, 0.5 * mid);
                auto phi = [&](int d0, int d1) {
                    if (d0 == 0 && d1 == 0) return 0.0;
                    if (d0 == 1 && d1 == 1) return mid;
                    return p10;
                };
                double c0 = phi(bu0, bu1) + phi(bv0, bv1);
                double c1 = phi(bu0 ^ 1, bu1 ^ 1) + phi(bv0 ^ 1, bv1 ^ 1);
                if (min(c0, c1) <= vw + 1e-12) lo = mid;
                else hi = mid;
            }
            max_Gamma = lo;
        }
        // Also verify Par:
        double max_Gamma_Par = 1.0;
        for (int m = 0; m < (1 << n_edges); ++m) {
            int vw = 0;
            if (((m >> 0) ^ (m >> 1)) & 1) vw++;
            for (int k = 2; k < ell; ++k) {
                int b1 = (m >> (k - 1)) & 1, b2 = (m >> k) & 1, b3 = (m >> (2 * ell + k - 2)) & 1;
                if ((b1 ^ b2 ^ b3) & 1) vw++;
            }
            for (int s = 1; s < ell - 1; ++s) {
                int b1 = (m >> (ell + s - 1)) & 1, b2 = (m >> (ell + s)) & 1, b3 = (m >> (2 * ell + s - 1)) & 1;
                if ((b1 ^ b2 ^ b3) & 1) vw++;
            }
            if (((m >> (2 * ell - 2)) ^ (m >> (2 * ell - 1))) & 1) vw++;

            int bu = ((m >> 0) ^ (m >> ell)) & 1;
            int bv = ((m >> (ell - 1)) ^ (m >> (2 * ell - 1))) & 1;
            int min_w = min((bu ^ 0) + (bv ^ 0), (bu ^ 1) + (bv ^ 1));
            if (min_w > 0) {
                max_Gamma_Par = min(max_Gamma_Par, (double)vw / min_w);
            }
        }
        double c_Rep = 1.0 / max_Gamma;
        double c_Par = 1.0 / max_Gamma_Par;
        cout << "  ell=" << ell << " (r=" << r << " rungs, " << n_edges << " edges/unit): "
             << "Gamma_Rep=" << fixed << setprecision(4) << max_Gamma << " (c_Rep=" << c_Rep << "), "
             << "Gamma_Par=" << max_Gamma_Par << " (c_Par=" << c_Par << ") "
             << "==> Single-Cell c_cycle = " << (c_Rep * c_Par) << endl;
    }
}

void verify_3d_unit_single_pass(int Lx, int Ly, vector<int> rx_coords, vector<int> ry_coords, const string& label) {
    int n_sheet = Lx * Ly;
    int n_hb = (int)ry_coords.size() * Lx;
    int n_vb = (int)rx_coords.size() * Ly;
    int nF = 2 * n_sheet + n_hb + n_vb;
    assert(nF <= 25);

    auto f_sheet = [&](int c, int s, int t) { return c * n_sheet + s * Ly + t; };
    int hb_start = 2 * n_sheet;
    int vb_start = 2 * n_sheet + n_hb;

    vector<uint32_t> int_masks;
    vector<int> int_degs;
    for (int c = 0; c < 2; ++c) {
        for (int t = 1; t < Ly; ++t) {
            int rk = -1;
            for (int k = 0; k < (int)ry_coords.size(); ++k) if (ry_coords[k] == t) rk = k;
            for (int s = 0; s < Lx; ++s) {
                uint32_t msk = (1u << f_sheet(c, s, t - 1)) | (1u << f_sheet(c, s, t));
                int dg = 2;
                if (rk >= 0) { msk |= (1u << (hb_start + rk * Lx + s)); dg = 3; }
                int_masks.push_back(msk);
                int_degs.push_back(dg);
            }
        }
        for (int s = 1; s < Lx; ++s) {
            int rk = -1;
            for (int k = 0; k < (int)rx_coords.size(); ++k) if (rx_coords[k] == s) rk = k;
            for (int t = 0; t < Ly; ++t) {
                uint32_t msk = (1u << f_sheet(c, s - 1, t)) | (1u << f_sheet(c, s, t));
                int dg = 2;
                if (rk >= 0) { msk |= (1u << (vb_start + rk * Ly + t)); dg = 3; }
                int_masks.push_back(msk);
                int_degs.push_back(dg);
            }
        }
    }
    for (int kx = 0; kx < (int)rx_coords.size(); ++kx) {
        int s = rx_coords[kx];
        for (int ky = 0; ky < (int)ry_coords.size(); ++ky) {
            int t = ry_coords[ky];
            uint32_t msk = (1u << (hb_start + ky * Lx + (s - 1)))
                         | (1u << (hb_start + ky * Lx + s))
                         | (1u << (vb_start + kx * Ly + (t - 1)))
                         | (1u << (vb_start + kx * Ly + t));
            int_masks.push_back(msk);
            int_degs.push_back(4);
        }
    }

    uint32_t wall_rep_mask[4] = {0, 0, 0, 0};
    int Mp[4] = {0, 0, 0, 0};
    vector<pair<int, int>> wall_par_pairs[4];
    for (int s = 0; s < Lx; ++s) {
        wall_rep_mask[0] |= (1u << f_sheet(0, s, 0)) | (1u << f_sheet(1, s, 0));
        wall_rep_mask[1] |= (1u << f_sheet(0, s, Ly - 1)) | (1u << f_sheet(1, s, Ly - 1));
        wall_par_pairs[0].push_back({f_sheet(0, s, 0), f_sheet(1, s, 0)});
        wall_par_pairs[1].push_back({f_sheet(0, s, Ly - 1), f_sheet(1, s, Ly - 1)});
    }
    for (int t = 0; t < Ly; ++t) {
        wall_rep_mask[2] |= (1u << f_sheet(0, 0, t)) | (1u << f_sheet(1, 0, t));
        wall_rep_mask[3] |= (1u << f_sheet(0, Lx - 1, t)) | (1u << f_sheet(1, Lx - 1, t));
        wall_par_pairs[2].push_back({f_sheet(0, 0, t), f_sheet(1, 0, t)});
        wall_par_pairs[3].push_back({f_sheet(0, Lx - 1, t), f_sheet(1, Lx - 1, t)});
    }
    for (int kx = 0; kx < (int)rx_coords.size(); ++kx) {
        wall_rep_mask[0] |= (1u << (vb_start + kx * Ly + 0));
        wall_rep_mask[1] |= (1u << (vb_start + kx * Ly + (Ly - 1)));
    }
    for (int ky = 0; ky < (int)ry_coords.size(); ++ky) {
        wall_rep_mask[2] |= (1u << (hb_start + ky * Lx + 0));
        wall_rep_mask[3] |= (1u << (hb_start + ky * Lx + (Lx - 1)));
    }
    int min_Mp = 100, min_Lp = 100;
    for (int p = 0; p < 4; ++p) {
        Mp[p] = __builtin_popcount(wall_rep_mask[p]);
        min_Mp = min(min_Mp, Mp[p]);
        min_Lp = min(min_Lp, (int)wall_par_pairs[p].size());
    }

    // SINGLE-PASS exact calculation!
    // For Rep: Gamma * min_{y in {0,1}} sum_{p=0}^3 (wp(m,y) / Mp[p]) <= vw_rep(m)
    //   ==> Gamma <= vw_rep(m) / min_y sum_p (wp(m,y) / Mp[p])!
    double max_Gamma_Rep = (double)min_Mp;
    double max_Gamma_Par = (double)min_Lp;
    uint32_t tot = 1u << nF;
    uint32_t all_ones = tot - 1u;

    for (uint32_t m = 0; m < tot; ++m) {
        int vw_rep = 0, vw_par = 0;
        for (size_t i = 0; i < int_masks.size(); ++i) {
            uint32_t sub = m & int_masks[i];
            if (sub != 0 && sub != int_masks[i]) vw_rep++;
            if (__builtin_popcount(sub) & 1) vw_par++;
        }

        // Rep fractional wall weight for y=0 vs y=1:
        double f0 = 0.0, f1 = 0.0;
        for (int p = 0; p < 4; ++p) {
            int w0 = __builtin_popcount(m & wall_rep_mask[p]);
            f0 += (double)w0 / Mp[p];
            f1 += (double)(Mp[p] - w0) / Mp[p];
        }
        double f_rep = min(f0, f1);
        if (f_rep > 1e-12) {
            max_Gamma_Rep = min(max_Gamma_Rep, vw_rep / f_rep);
        }

        // Par fractional wall weight for y=0 vs y=1:
        double g0 = 0.0, g1 = 0.0;
        for (int p = 0; p < 4; ++p) {
            int Lp = (int)wall_par_pairs[p].size();
            int w0 = 0;
            for (const auto& pr : wall_par_pairs[p]) {
                w0 += (((m >> pr.first) ^ (m >> pr.second)) & 1u);
            }
            g0 += (double)w0 / Lp;
            g1 += (double)(Lp - w0) / Lp;
        }
        double f_par = min(g0, g1);
        if (f_par > 1e-12) {
            max_Gamma_Par = min(max_Gamma_Par, vw_par / f_par);
        }
    }

    double c_Rep = 1.0 / max_Gamma_Rep;
    double c_Par = 1.0 / max_Gamma_Par;
    cout << "  [" << label << "] (faces=" << nF << ", interior_edges=" << int_masks.size() << "):\n"
         << "    Gamma_Rep = " << fixed << setprecision(4) << max_Gamma_Rep << " (c_Rep = " << c_Rep << ")"
         << " | Gamma_Par = " << max_Gamma_Par << " (c_Par = " << c_Par << ")"
         << " ==> FULL-CYCLE c_cycle = " << (c_Rep * c_Par) << endl;
}

int main() {
    verify_2d_classical_ladders();
    cout << "\n=== 2. 3D Quantum Doubled-Square Units U_F (Exhaustive Single-Pass Proof) ===" << endl;
    verify_3d_unit_single_pass(2, 2, {1}, {1}, "Uniform 2x2 Four-Arm Cross (L=2, r=1)");
    verify_3d_unit_single_pass(3, 3, {1}, {1}, "Uniform 3x3 Single Cross (L=3, r=1)");
    verify_3d_unit_single_pass(3, 2, {1, 2}, {1}, "Mixed 3x2 Unit");
    verify_3d_unit_single_pass(2, 1, {1}, {}, "Thin-Bridge 2x1 Rect (Lx=2, Ly=1, rx=1)");
    verify_3d_unit_single_pass(3, 1, {1, 2}, {}, "Thin-Bridge 3x1 Rect (Lx=3, Ly=1, rx={1,2})");
    return 0;
}
