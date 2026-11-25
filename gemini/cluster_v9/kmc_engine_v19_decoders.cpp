// kmc_engine_v19_decoders.cpp
// Multithreaded C++20 BKL Thermal Engine + Paired 4-Decoder Single-Shot Readout Suite:
//   Evaluates FOUR distinct classical decoders on the EXACT SAME physical BKL trajectories e(t_g):
//     1. `uf`      : Standard unweighted global Union-Find + local GF(2) elimination (paper baseline)
//     2. `cool_uf` : Zero-temperature (T=0, dE < 0) greedy energy-descent cooling pre-pass + UF
//     3. `hier_uf` : Multi-scale local-to-global cluster UF (staged caps R = 16 -> 64 -> 256 -> inf)
//     4. `bp_osd`  : Damped Normalized Min-Sum Belief Propagation (alpha=0.75, damping=0.50, 28 iters)
//                    + Reliability-Ordered OSD-CS(36) (full-matrix for N <= 3200; soft-priority
//                    cluster growth + cluster OSD-CS(36) up to 1600 columns for N > 3200).
//   Also provides `verify`, `prep_logicals`, and `self_test` modes with hard GF(2) syndrome gates.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct BitRow {
    std::vector<uint64_t> w;
    BitRow() = default;
    explicit BitRow(int nbits) : w((nbits + 63) / 64, 0ULL) {}
    inline void flip(int b) { w[b >> 6] ^= (1ULL << (b & 63)); }
    inline bool get(int b) const { return (w[b >> 6] >> (b & 63)) & 1ULL; }
    inline void xor_with(const BitRow& o) {
        const size_t n = w.size();
        for (size_t i = 0; i < n; ++i) w[i] ^= o.w[i];
    }
    inline void xor_from(const BitRow& o, size_t w0) {
        const size_t n = w.size();
        for (size_t i = w0; i < n; ++i) w[i] ^= o.w[i];
    }
    inline int popcount() const {
        int s = 0;
        for (uint64_t x : w) s += __builtin_popcountll(x);
        return s;
    }
    inline int first_one() const {
        for (size_t i = 0; i < w.size(); ++i) {
            if (w[i]) return (int)(i * 64 + __builtin_ctzll(w[i]));
        }
        return -1;
    }
};

struct Code {
    int N = 0, M_check = 0, M_stab = 0, K = 0, max_q_deg = 0;
    std::vector<std::vector<int>> checks, q_to_checks, stabs, logicals;
    std::vector<int> j_tenths;

    static Code load(const std::string& path) {
        std::ifstream in(path);
        if (!in) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
        Code c;
        in >> c.N >> c.M_check >> c.M_stab >> c.K;
        c.checks.resize(c.M_check);
        c.q_to_checks.resize(c.N);
        for (int i = 0; i < c.M_check; ++i) {
            int d; in >> d; c.checks[i].resize(d);
            for (int j = 0; j < d; ++j) {
                in >> c.checks[i][j];
                c.q_to_checks[c.checks[i][j]].push_back(i);
            }
        }
        for (int q = 0; q < c.N; ++q) {
            c.max_q_deg = std::max(c.max_q_deg, (int)c.q_to_checks[q].size());
        }
        c.stabs.resize(c.M_stab);
        for (int i = 0; i < c.M_stab; ++i) {
            int d; in >> d; c.stabs[i].resize(d);
            for (int j = 0; j < d; ++j) in >> c.stabs[i][j];
        }
        c.logicals.resize(c.K);
        for (int i = 0; i < c.K; ++i) {
            int d; in >> d; c.logicals[i].resize(d);
            for (int j = 0; j < d; ++j) in >> c.logicals[i][j];
        }
        c.j_tenths.assign(c.M_check, 10);
        return c;
    }

    inline int verify_commutation() const {
        std::vector<uint8_t> c_par(M_check, 0);
        std::vector<int> touched;
        int bad = 0;
        for (const auto& st : stabs) {
            touched.clear();
            for (int q : st) {
                for (int ci : q_to_checks[q]) {
                    if (!c_par[ci]) touched.push_back(ci);
                    c_par[ci] ^= 1;
                }
            }
            for (int ci : touched) {
                if (c_par[ci]) ++bad;
                c_par[ci] = 0;
            }
        }
        std::vector<uint8_t> lmark(N, 0);
        for (const auto& lg : logicals) {
            for (int q : lg) lmark[q] = 1;
            for (const auto& st : stabs) {
                int p = 0;
                for (int q : st) p ^= lmark[q];
                if (p) ++bad;
            }
            for (int q : lg) lmark[q] = 0;
        }
        return bad;
    }

    inline bool has_logical_error(const std::vector<uint8_t>& e, const std::vector<uint8_t>& corr) const {
        for (int a = 0; a < K; ++a) {
            uint8_t p = 0;
            for (int q : logicals[a]) p ^= (e[q] ^ corr[q]);
            if (p) return true;
        }
        return false;
    }

    inline bool has_logical_error_raw(const std::vector<uint8_t>& e) const {
        for (int a = 0; a < K; ++a) {
            uint8_t p = 0;
            for (int q : logicals[a]) p ^= e[q];
            if (p) return true;
        }
        return false;
    }

    inline bool check_syndrome_match(const std::vector<uint8_t>& s, const std::vector<uint8_t>& corr) const {
        for (int c = 0; c < M_check; ++c) {
            uint8_t p = 0;
            for (int q : checks[c]) p ^= corr[q];
            if (p != s[c]) return false;
        }
        return true;
    }
};

// =========================================================================================
// Exact GF(2) Submatrix & Ordered-Statistics (OSD-CS) Solvers
// =========================================================================================
bool solve_gf2_dense_core(
    int nr, int nc,
    const std::vector<std::vector<int>>& row_cols,
    const std::vector<uint8_t>& rhs,
    std::vector<uint8_t>& sol_out,
    int max_refine_nc = 380
) {
    int width = nc + 1;
    std::vector<BitRow> mat(nr, BitRow(width));
    for (int r = 0; r < nr; ++r) {
        for (int c : row_cols[r]) mat[r].flip(c);
        if (rhs[r]) mat[r].flip(nc);
    }
    std::vector<int> pcol(nr, -1), prow(nc, -1);
    int rank = 0;
    bool do_refine = (max_refine_nc > 0) && (nc <= max_refine_nc);
    for (int c = 0; c < nc && rank < nr; ++c) {
        int sel = -1;
        for (int r = rank; r < nr; ++r) if (mat[r].get(c)) { sel = r; break; }
        if (sel == -1) continue;
        if (sel != rank) std::swap(mat[sel], mat[rank]);
        pcol[rank] = c; prow[c] = rank;
        size_t w0 = (size_t)(c >> 6);
        if (do_refine) {
            for (int r = 0; r < nr; ++r) if (r != rank && mat[r].get(c)) mat[r].xor_from(mat[rank], w0);
        } else {
            for (int r = rank + 1; r < nr; ++r) if (mat[r].get(c)) mat[r].xor_from(mat[rank], w0);
        }
        ++rank;
    }
    for (int r = rank; r < nr; ++r) if (mat[r].get(nc)) return false;

    sol_out.assign(nc, 0);
    if (do_refine) {
        for (int r = 0; r < rank; ++r) if (mat[r].get(nc)) sol_out[pcol[r]] = 1;
        if (nc > rank) {
            BitRow cur(nc);
            for (int c = 0; c < nc; ++c) if (sol_out[c]) cur.flip(c);
            int cur_w = cur.popcount();
            std::vector<BitRow> nvs;
            for (int c = 0; c < nc && (int)nvs.size() < 40; ++c) {
                if (prow[c] != -1) continue;
                BitRow nv(nc); nv.flip(c);
                for (int r = 0; r < rank; ++r) if (mat[r].get(c)) nv.flip(pcol[r]);
                nvs.push_back(std::move(nv));
            }
            bool imp = true; int passes = 0;
            while (imp && passes < 4 && cur_w > 1) {
                imp = false; ++passes;
                for (const auto& nv : nvs) {
                    int w2 = 0;
                    for (size_t i = 0; i < cur.w.size(); ++i) w2 += __builtin_popcountll(cur.w[i] ^ nv.w[i]);
                    if (w2 < cur_w) { cur.xor_with(nv); cur_w = w2; imp = true; }
                }
            }
            for (int c = 0; c < nc; ++c) sol_out[c] = cur.get(c) ? 1 : 0;
        }
    } else {
        BitRow cur_sol(nc + 1);
        for (int r = rank - 1; r >= 0; --r) {
            int pc = pcol[r];
            uint8_t val = mat[r].get(nc) ? 1 : 0;
            size_t w0 = (size_t)(pc >> 6);
            int parity = 0;
            for (size_t wi = w0; wi < cur_sol.w.size(); ++wi) {
                parity ^= (__builtin_popcountll(mat[r].w[wi] & cur_sol.w[wi]) & 1);
            }
            val ^= (uint8_t)parity;
            if (val) {
                sol_out[pc] = 1;
                cur_sol.flip(pc);
            }
        }
    }
    return true;
}

bool solve_gf2_submatrix(
    int nr, int nc,
    const std::vector<std::vector<int>>& row_cols,
    const std::vector<uint8_t>& rhs,
    std::vector<uint8_t>& sol_out,
    int max_refine_nc = 380
) {
    if (nc <= std::max(380, max_refine_nc)) {
        return solve_gf2_dense_core(nr, nc, row_cols, rhs, sol_out, max_refine_nc);
    }
    std::vector<std::vector<int>> r_cols(nr);
    std::vector<uint8_t> r_rhs = rhs, elim(nr, 0);
    std::vector<std::vector<int>> c_rows(nc);
    std::vector<int> col_deg(nc, 0);
    const int MAX_B = 64;
    std::vector<int> bkt[MAX_B + 2];

    for (int r = 0; r < nr; ++r) {
        r_cols[r] = row_cols[r];
        std::sort(r_cols[r].begin(), r_cols[r].end());
        for (int c : r_cols[r]) {
            c_rows[c].push_back(r);
            ++col_deg[c];
        }
        int b = std::min(MAX_B + 1, (int)r_cols[r].size());
        bkt[b].push_back(r);
    }

    int active = nr;
    int cur_b = 0;
    std::vector<std::pair<int, int>> piv_stack;
    piv_stack.reserve(nr);
    sol_out.assign(nc, 0);

    auto xor_sorted = [](const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) {
        out.clear();
        out.reserve(a.size() + b.size());
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] < b[j]) out.push_back(a[i++]);
            else if (b[j] < a[i]) out.push_back(b[j++]);
            else { ++i; ++j; }
        }
        while (i < a.size()) out.push_back(a[i++]);
        while (j < b.size()) out.push_back(b[j++]);
    };

    std::vector<int> merged;
    while (active > 0) {
        while (!bkt[0].empty()) {
            int r = bkt[0].back();
            bkt[0].pop_back();
            if (elim[r] || !r_cols[r].empty()) continue;
            if (r_rhs[r]) return false;
            elim[r] = 1;
            --active;
        }
        if (active == 0) break;

        int sel_r = -1;
        while (cur_b <= MAX_B + 1) {
            while (!bkt[cur_b].empty()) {
                int r = bkt[cur_b].back();
                bkt[cur_b].pop_back();
                if (elim[r]) continue;
                int b_now = std::min(MAX_B + 1, (int)r_cols[r].size());
                if (b_now != cur_b) {
                    bkt[b_now].push_back(r);
                    if (b_now < cur_b) cur_b = b_now;
                    continue;
                }
                sel_r = r;
                break;
            }
            if (sel_r != -1) break;
            ++cur_b;
        }
        if (sel_r == -1) break;

        if ((cur_b >= 20 && active <= 4500) || (cur_b >= MAX_B + 1 && active <= 7500)) {
            std::vector<int> rem_rows;
            rem_rows.reserve(active);
            std::vector<int> col_map(nc, -1), sub_cols;
            for (int r = 0; r < nr; ++r) {
                if (elim[r]) continue;
                rem_rows.push_back(r);
                for (int c : r_cols[r]) {
                    if (col_map[c] == -1) {
                        col_map[c] = (int)sub_cols.size();
                        sub_cols.push_back(c);
                    }
                }
            }
            int sub_nr = (int)rem_rows.size(), sub_nc = (int)sub_cols.size();
            std::vector<std::vector<int>> sub_rc(sub_nr);
            std::vector<uint8_t> sub_rhs(sub_nr, 0), sub_sol;
            for (int i = 0; i < sub_nr; ++i) {
                int r = rem_rows[i];
                sub_rhs[i] = r_rhs[r];
                sub_rc[i].reserve(r_cols[r].size());
                for (int c : r_cols[r]) sub_rc[i].push_back(col_map[c]);
            }
            if (!solve_gf2_dense_core(sub_nr, sub_nc, sub_rc, sub_rhs, sub_sol, 0)) return false;
            for (int j = 0; j < sub_nc; ++j) sol_out[sub_cols[j]] = sub_sol[j];
            break;
        }

        int pc = r_cols[sel_r][0];
        int best_deg = col_deg[pc];
        for (int c : r_cols[sel_r]) {
            if (col_deg[c] < best_deg) { best_deg = col_deg[c]; pc = c; }
        }
        elim[sel_r] = 1;
        --active;
        piv_stack.push_back({sel_r, pc});
        for (int c : r_cols[sel_r]) --col_deg[c];

        std::vector<int> targets = std::move(c_rows[pc]);
        c_rows[pc].clear();
        for (int r2 : targets) {
            if (elim[r2]) continue;
            if (!std::binary_search(r_cols[r2].begin(), r_cols[r2].end(), pc)) continue;
            for (int c : r_cols[r2]) --col_deg[c];
            xor_sorted(r_cols[r2], r_cols[sel_r], merged);
            for (int c : merged) {
                ++col_deg[c];
                if (!std::binary_search(r_cols[r2].begin(), r_cols[r2].end(), c)) {
                    c_rows[c].push_back(r2);
                }
            }
            r_cols[r2] = merged;
            r_rhs[r2] ^= r_rhs[sel_r];
            int b2 = std::min(MAX_B + 1, (int)r_cols[r2].size());
            bkt[b2].push_back(r2);
            if (b2 < cur_b) cur_b = b2;
        }
    }

    for (int idx = (int)piv_stack.size() - 1; idx >= 0; --idx) {
        int r = piv_stack[idx].first, pc = piv_stack[idx].second;
        uint8_t v = r_rhs[r];
        for (int c : r_cols[r]) if (c != pc) v ^= sol_out[c];
        sol_out[pc] = v;
    }
    return true;
}

// =========================================================================================
// Reliability-Ordered OSD-CS(36) Solver (columns ALREADY ordered from lowest L_q to highest L_q!)
// Minimizes smooth positive soft BP cost sum_{c: sol[c]==1} soft_cost[c] via RREF OSD-0 + CS(36, pair=18).
// =========================================================================================
bool solve_osd_cs_ordered(
    int nr, int nc,
    const std::vector<std::vector<int>>& row_cols,
    const std::vector<uint8_t>& rhs,
    const std::vector<float>& col_soft_cost,
    std::vector<uint8_t>& sol_out,
    int max_rref_nc = 2200
) {
    if (nc > max_rref_nc) {
        // Columns are ALREADY ordered by ascending reliability L_q; run sparse solver directly!
        return solve_gf2_submatrix(nr, nc, row_cols, rhs, sol_out, 0);
    }
    int width = nc + 1;
    std::vector<BitRow> mat(nr, BitRow(width));
    for (int r = 0; r < nr; ++r) {
        for (int c : row_cols[r]) mat[r].flip(c);
        if (rhs[r]) mat[r].flip(nc);
    }
    std::vector<int> pcol(nr, -1), prow(nc, -1);
    int rank = 0;
    for (int c = 0; c < nc && rank < nr; ++c) {
        int sel = -1;
        for (int r = rank; r < nr; ++r) if (mat[r].get(c)) { sel = r; break; }
        if (sel == -1) continue;
        if (sel != rank) std::swap(mat[sel], mat[rank]);
        pcol[rank] = c; prow[c] = rank;
        size_t w0 = (size_t)(c >> 6);
        for (int r = 0; r < nr; ++r) {
            if (r != rank && mat[r].get(c)) mat[r].xor_from(mat[rank], w0);
        }
        ++rank;
    }
    for (int r = rank; r < nr; ++r) if (mat[r].get(nc)) return false;

    // OSD-0 base solution (pivots lie on lowest-LLR columns; all free reliable columns = 0):
    BitRow base(nc);
    for (int r = 0; r < rank; ++r) {
        if (mat[r].get(nc)) base.flip(pcol[r]);
    }

    auto eval_cost = [&](const BitRow& v) -> double {
        double tot = 0.0;
        for (size_t wi = 0; wi < v.w.size(); ++wi) {
            uint64_t x = v.w[wi];
            while (x) {
                int bit = __builtin_ctzll(x);
                tot += (double)col_soft_cost[(int)(wi * 64 + bit)];
                x &= (x - 1ULL);
            }
        }
        return tot;
    };

    BitRow best = base;
    double best_cost = eval_cost(base);

    // Extract up to 36 least-reliable non-pivot nullspace vectors:
    std::vector<BitRow> nvs;
    nvs.reserve(36);
    for (int c = 0; c < nc && (int)nvs.size() < 36; ++c) {
        if (prow[c] != -1) continue;
        BitRow nv(nc);
        nv.flip(c);
        for (int r = 0; r < rank; ++r) {
            if (mat[r].get(c)) nv.flip(pcol[r]);
        }
        nvs.push_back(std::move(nv));
    }

    int K_nv = (int)nvs.size();
    BitRow cand(nc);
    // Order-1 Combination Sweep across all K_nv <= 36 least-reliable free columns:
    for (int i = 0; i < K_nv; ++i) {
        cand = base;
        cand.xor_with(nvs[i]);
        double c1 = eval_cost(cand);
        if (c1 + 1e-9 < best_cost) {
            best = cand;
            best_cost = c1;
        }
    }
    // Order-2 Combination Sweep across top 16 least-reliable free columns (120 pairs):
    int K2 = std::min(K_nv, 16);
    for (int i = 0; i < K2; ++i) {
        for (int j = i + 1; j < K2; ++j) {
            cand = base;
            cand.xor_with(nvs[i]);
            cand.xor_with(nvs[j]);
            double c2 = eval_cost(cand);
            if (c2 + 1e-9 < best_cost) {
                best = cand;
                best_cost = c2;
            }
        }
    }
    // Greedy coordinate polish from best found solution:
    for (int pass = 0; pass < 2; ++pass) {
        bool imp = false;
        for (int i = 0; i < K_nv; ++i) {
            cand = best;
            cand.xor_with(nvs[i]);
            double cc = eval_cost(cand);
            if (cc + 1e-9 < best_cost) {
                best = cand;
                best_cost = cc;
                imp = true;
            }
        }
        if (!imp) break;
    }

    sol_out.assign(nc, 0);
    for (int c = 0; c < nc; ++c) sol_out[c] = best.get(c) ? 1 : 0;
    return true;
}

// =========================================================================================
// Decoders 1, 2, 3: UF (`use_maj=false`), Cool-UF (`use_maj=true`), and Hier-UF
// =========================================================================================
struct FastDecoderUF {
    const Code& code;
    int V;
    mutable std::vector<int> parent, global_q_to_local, touched_nodes, seen_stamp;
    mutable std::vector<uint8_t> cluster_solved, res_s, work_s, in_work_act, in_cand_q, in_dirty_corr;
    mutable std::vector<int> dirty_corr, work_act, cand_q, touched_work_c;
    mutable std::vector<std::vector<int>> cl_checks, cl_qubits, cl_f_checks, cl_f_qubits;
    mutable int cur_stamp = 1;

    explicit FastDecoderUF(const Code& c)
        : code(c), V(c.M_check + c.N), parent(V, -1), global_q_to_local(c.N, -1),
          seen_stamp(V, 0), cluster_solved(V, 0), res_s(c.M_check, 0), work_s(c.M_check, 0),
          in_work_act(c.M_check, 0), in_cand_q(c.N, 0), in_dirty_corr(c.N, 0),
          cl_checks(V), cl_qubits(V), cl_f_checks(V), cl_f_qubits(V) {}

    inline void clear_corr(std::vector<uint8_t>& corr_out) const {
        for (int q : dirty_corr) { corr_out[q] = 0; in_dirty_corr[q] = 0; }
        dirty_corr.clear();
    }

    inline void flip_corr(int q, std::vector<uint8_t>& corr_out) const {
        if (!in_dirty_corr[q]) { in_dirty_corr[q] = 1; dirty_corr.push_back(q); }
        corr_out[q] ^= 1;
    }

    inline int find_root(int x) const {
        int r = x;
        while (parent[r] != r) r = parent[r];
        while (x != r) { int nxt = parent[x]; parent[x] = r; x = nxt; }
        return r;
    }
    inline void touch(int node) const { if (parent[node] == -1) touched_nodes.push_back(node); }

    int merge_clusters(int a, int b, int max_local_qubits = 0) const {
        a = find_root(a); b = find_root(b);
        if (a == b) return a;
        if (cl_checks[a].size() + cl_qubits[a].size() < cl_checks[b].size() + cl_qubits[b].size()) std::swap(a, b);
        parent[b] = a;
        cl_checks[a].insert(cl_checks[a].end(), cl_checks[b].begin(), cl_checks[b].end());
        cl_qubits[a].insert(cl_qubits[a].end(), cl_qubits[b].begin(), cl_qubits[b].end());
        cl_f_checks[a].insert(cl_f_checks[a].end(), cl_f_checks[b].begin(), cl_f_checks[b].end());
        cl_f_qubits[a].insert(cl_f_qubits[a].end(), cl_f_qubits[b].begin(), cl_f_qubits[b].end());
        cl_checks[b].clear(); cl_qubits[b].clear(); cl_f_checks[b].clear(); cl_f_qubits[b].clear();
        cluster_solved[a] = (max_local_qubits > 0 && (int)cl_qubits[a].size() > max_local_qubits) ? 1 : 0;
        return a;
    }

    void run_uf_core(const std::vector<int>& active_checks, const std::vector<uint8_t>& s_in,
                     std::vector<uint8_t>& corr_out, int max_local_qubits = 0, int max_refine_nc = 380) const {
        if (active_checks.empty()) return;
        touched_nodes.clear();
        for (int c : active_checks) {
            res_s[c] = s_in[c]; touch(c); parent[c] = c; cluster_solved[c] = 0;
            cl_checks[c].push_back(c); cl_f_checks[c].push_back(c);
        }
        std::vector<int> unsolved, cand_roots;
        while (true) {
            ++cur_stamp; unsolved.clear();
            for (int c : active_checks) {
                if (!res_s[c]) continue;
                int r = find_root(c);
                if (!cluster_solved[r] && seen_stamp[r] != cur_stamp) {
                    seen_stamp[r] = cur_stamp; unsolved.push_back(r);
                }
            }
            if (unsolved.empty()) break;
            for (int r0 : unsolved) {
                int r = find_root(r0);
                std::vector<int> fc = std::move(cl_f_checks[r]); cl_f_checks[r].clear();
                for (int c : fc) {
                    for (int q : code.checks[c]) {
                        int qnode = code.M_check + q;
                        if (parent[qnode] == -1) {
                            touch(qnode); parent[qnode] = r;
                            cl_qubits[r].push_back(q); cl_f_qubits[r].push_back(q);
                        } else r = merge_clusters(r, qnode, max_local_qubits);
                    }
                }
            }
            ++cur_stamp; cand_roots.clear();
            for (int c : active_checks) {
                if (!res_s[c]) continue;
                int r = find_root(c);
                if (!cluster_solved[r] && seen_stamp[r] != cur_stamp) { seen_stamp[r] = cur_stamp; cand_roots.push_back(r); }
            }
            for (int r0 : cand_roots) {
                int r = find_root(r0);
                std::vector<int> fq = std::move(cl_f_qubits[r]); cl_f_qubits[r].clear();
                for (int q : fq) {
                    for (int c : code.q_to_checks[q]) {
                        if (parent[c] == -1) {
                            touch(c); parent[c] = r; res_s[c] = s_in[c];
                            cl_checks[r].push_back(c); cl_f_checks[r].push_back(c);
                        } else r = merge_clusters(r, c, max_local_qubits);
                    }
                }
            }
            ++cur_stamp; cand_roots.clear();
            for (int c : active_checks) {
                if (!res_s[c]) continue;
                int r = find_root(c);
                if (!cluster_solved[r] && seen_stamp[r] != cur_stamp) { seen_stamp[r] = cur_stamp; cand_roots.push_back(r); }
            }
            for (int r : cand_roots) {
                auto& qs = cl_qubits[r];
                const auto& cs = cl_checks[r];
                if (qs.empty()) continue;
                if (max_local_qubits > 0 && (int)qs.size() > max_local_qubits) {
                    cluster_solved[r] = 1;
                    continue;
                }
                int n_act_r = 0;
                for (int ci : cs) if (res_s[ci]) ++n_act_r;
                if (n_act_r == 0) { cluster_solved[r] = 1; continue; }

                int match_q1 = -1;
                if (n_act_r <= code.max_q_deg) {
                    for (int q : qs) {
                        const auto& qc = code.q_to_checks[q];
                        if ((int)qc.size() == n_act_r) {
                            bool all_on = true;
                            for (int ci : qc) if (!res_s[ci]) { all_on = false; break; }
                            if (all_on) { match_q1 = q; break; }
                        }
                    }
                }
                if (match_q1 != -1) {
                    flip_corr(match_q1, corr_out);
                    for (int ci : code.q_to_checks[match_q1]) res_s[ci] ^= 1;
                    cluster_solved[r] = 1;
                    continue;
                }

                int nc = (int)qs.size(), nr = (int)cs.size();
                for (int j = 0; j < nc; ++j) global_q_to_local[qs[j]] = j;
                std::vector<std::vector<int>> rcols(nr);
                std::vector<uint8_t> rhs(nr, 0);
                for (int i = 0; i < nr; ++i) {
                    int ci = cs[i]; rhs[i] = res_s[ci];
                    for (int q : code.checks[ci]) {
                        int loc = global_q_to_local[q];
                        if (loc != -1) rcols[i].push_back(loc);
                    }
                }
                for (int j = 0; j < nc; ++j) global_q_to_local[qs[j]] = -1;
                std::vector<uint8_t> lsol;
                if (solve_gf2_submatrix(nr, nc, rcols, rhs, lsol, max_refine_nc)) {
                    for (int j = 0; j < nc; ++j) {
                        if (lsol[j]) {
                            int q = qs[j]; flip_corr(q, corr_out);
                            for (int ci : code.q_to_checks[q]) res_s[ci] ^= 1;
                        }
                    }
                    cluster_solved[r] = 1;
                }
            }
        }
        for (int node : touched_nodes) {
            if (node < code.M_check) res_s[node] = 0;
            parent[node] = -1; cluster_solved[node] = 0;
            cl_checks[node].clear(); cl_qubits[node].clear();
            cl_f_checks[node].clear(); cl_f_qubits[node].clear();
        }
    }

    // 1. Baseline UF:
    void decode_uf(const std::vector<int>& active_checks, const std::vector<uint8_t>& s_in,
                   std::vector<uint8_t>& corr_out) const {
        clear_corr(corr_out);
        if (active_checks.empty()) return;
        run_uf_core(active_checks, s_in, corr_out, 0, 380);
    }

    // 2. Cool-UF (T=0 greedy strictly-downhill energy descent + UF; sets cooled_to_zero=true if T=0 quench reaches s==0!):
    void decode_cool_uf(const std::vector<int>& active_checks,
                        std::vector<uint8_t>& corr_out,
                        bool& cooled_to_zero) const {
        clear_corr(corr_out);
        cooled_to_zero = true;
        if (active_checks.empty()) return;
        touched_work_c.clear();
        std::vector<int> pq[16];
        int top_g = 0;
        auto push_cand = [&](int q) {
            const auto& qc = code.q_to_checks[q];
            int on = 0;
            for (int c : qc) on += work_s[c];
            int g = 2 * on - (int)qc.size();
            if (g > 0) {
                if (g > 15) g = 15;
                if (g > (int)in_cand_q[q]) {
                    in_cand_q[q] = (uint8_t)g;
                    pq[g].push_back(q);
                    if (g > top_g) top_g = g;
                }
            }
        };
        cand_q.clear();
        for (int c : active_checks) {
            work_s[c] = 1; touched_work_c.push_back(c);
            for (int q : code.checks[c]) {
                if (!in_cand_q[q]) { in_cand_q[q] = 255; cand_q.push_back(q); }
            }
        }
        for (int q : cand_q) { in_cand_q[q] = 0; push_cand(q); }
        while (top_g > 0) {
            if (pq[top_g].empty()) { --top_g; continue; }
            int q = pq[top_g].back();
            pq[top_g].pop_back();
            if ((int)in_cand_q[q] > top_g) continue;
            in_cand_q[q] = 0;
            const auto& qc = code.q_to_checks[q];
            int on = 0;
            for (int c : qc) on += work_s[c];
            int cur_g = 2 * on - (int)qc.size();
            if (cur_g <= 0) continue;
            if (cur_g > 15) cur_g = 15;
            if (cur_g < top_g) {
                in_cand_q[q] = (uint8_t)cur_g;
                pq[cur_g].push_back(q);
                continue;
            }
            flip_corr(q, corr_out);
            for (int c : qc) {
                if (!work_s[c]) {
                    work_s[c] = 1; touched_work_c.push_back(c);
                } else {
                    work_s[c] = 0;
                }
                for (int q2 : code.checks[c]) push_cand(q2);
            }
        }
        work_act.clear();
        for (int c : touched_work_c) {
            if (work_s[c] && !in_work_act[c]) { in_work_act[c] = 1; work_act.push_back(c); }
        }
        if (!work_act.empty()) {
            cooled_to_zero = false;
            run_uf_core(work_act, work_s, corr_out, 0, 380);
        }
        for (int c : touched_work_c) { work_s[c] = 0; in_work_act[c] = 0; }
    }

    // 3. Hier-UF (Multi-scale local-to-global cluster UF: R = 16 -> 64 -> 256 -> inf):
    void decode_hier_uf(const std::vector<int>& active_checks,
                        const FastDecoderUF& stage_helper,
                        std::vector<uint8_t>& stage_buf,
                        std::vector<uint8_t>& corr_out) const {
        clear_corr(corr_out);
        if (active_checks.empty()) return;
        touched_work_c.clear();
        for (int c : active_checks) {
            work_s[c] = 1;
            in_work_act[c] = 1;
            touched_work_c.push_back(c);
        }
        work_act = active_checks;

        const int caps[4] = {16, 64, 256, 0};
        for (int st = 0; st < 4 && !work_act.empty(); ++st) {
            stage_helper.clear_corr(stage_buf);
            stage_helper.run_uf_core(work_act, work_s, stage_buf, caps[st], 1024);
            for (int q : stage_helper.dirty_corr) {
                if (stage_buf[q]) {
                    flip_corr(q, corr_out);
                    for (int c : code.q_to_checks[q]) {
                        if (!in_work_act[c]) { in_work_act[c] = 1; touched_work_c.push_back(c); }
                        work_s[c] ^= 1;
                    }
                }
            }
            work_act.clear();
            for (int c : touched_work_c) {
                if (work_s[c]) work_act.push_back(c);
            }
        }
        stage_helper.clear_corr(stage_buf);
        for (int c : touched_work_c) { work_s[c] = 0; in_work_act[c] = 0; }
    }
};

// =========================================================================================
// Decoder 4: Damped Normalized Min-Sum BP + Reliability-Ordered OSD-CS(36) (`BPOSDDecoder`)
// =========================================================================================
struct BPOSDDecoder {
    const Code& code;
    int E = 0;
    // Flat CSR edge indexing for cache-contiguous BP message passing:
    std::vector<int> c_off, c_edges, c_q;
    std::vector<int> q_off, q_edges;

    mutable std::vector<float> msg_q2c, msg_c2q, post_L, soft_cost;
    mutable std::vector<uint8_t> in_sub_c, in_sub_q, in_dirty_corr;
    mutable std::vector<int> sub_c, sub_q, dirty_corr;

    // Soft-cluster UF buffers for N > 3200:
    int V;
    mutable std::vector<int> parent, global_q_to_local, touched_nodes, seen_stamp;
    mutable std::vector<uint8_t> cluster_solved, res_s;
    mutable std::vector<std::vector<int>> cl_checks, cl_qubits, cl_f_checks, cl_cand_q;
    mutable int cur_stamp = 1;

    explicit BPOSDDecoder(const Code& c)
        : code(c), c_off(c.M_check + 1, 0), q_off(c.N + 1, 0),
          post_L(c.N, 4.0f), soft_cost(c.N, 8.0f),
          in_sub_c(c.M_check, 0), in_sub_q(c.N, 0), in_dirty_corr(c.N, 0),
          V(c.M_check + c.N), parent(V, -1), global_q_to_local(c.N, -1),
          seen_stamp(V, 0), cluster_solved(V, 0), res_s(c.M_check, 0),
          cl_checks(V), cl_qubits(V), cl_f_checks(V), cl_cand_q(V) {
        for (int ci = 0; ci < c.M_check; ++ci) c_off[ci + 1] = c_off[ci] + (int)c.checks[ci].size();
        E = c_off[c.M_check];
        c_q.resize(E);
        c_edges.resize(E);
        for (int ci = 0; ci < c.M_check; ++ci) {
            for (size_t j = 0; j < c.checks[ci].size(); ++j) {
                int e_id = c_off[ci] + (int)j;
                int q = c.checks[ci][j];
                c_q[e_id] = q;
                c_edges[e_id] = e_id;
                ++q_off[q + 1];
            }
        }
        for (int q = 0; q < c.N; ++q) q_off[q + 1] += q_off[q];
        q_edges.resize(E);
        std::vector<int> cur_q = q_off;
        for (int e_id = 0; e_id < E; ++e_id) {
            int q = c_q[e_id];
            q_edges[cur_q[q]++] = e_id;
        }
        msg_q2c.assign(E, 1.0f);
        msg_c2q.assign(E, 0.0f);
    }

    inline void clear_corr(std::vector<uint8_t>& corr_out) const {
        for (int q : dirty_corr) { corr_out[q] = 0; in_dirty_corr[q] = 0; }
        dirty_corr.clear();
    }

    inline void flip_corr(int q, std::vector<uint8_t>& corr_out) const {
        if (!in_dirty_corr[q]) { in_dirty_corr[q] = 1; dirty_corr.push_back(q); }
        corr_out[q] ^= 1;
    }

    inline int find_root(int x) const {
        int r = x;
        while (parent[r] != r) r = parent[r];
        while (x != r) { int nxt = parent[x]; parent[x] = r; x = nxt; }
        return r;
    }
    inline void touch(int node) const { if (parent[node] == -1) touched_nodes.push_back(node); }

    int merge_soft_clusters(int a, int b) const {
        a = find_root(a); b = find_root(b);
        if (a == b) return a;
        if (cl_checks[a].size() + cl_qubits[a].size() < cl_checks[b].size() + cl_qubits[b].size()) std::swap(a, b);
        parent[b] = a;
        cl_checks[a].insert(cl_checks[a].end(), cl_checks[b].begin(), cl_checks[b].end());
        cl_qubits[a].insert(cl_qubits[a].end(), cl_qubits[b].begin(), cl_qubits[b].end());
        cl_f_checks[a].insert(cl_f_checks[a].end(), cl_f_checks[b].begin(), cl_f_checks[b].end());
        cl_cand_q[a].insert(cl_cand_q[a].end(), cl_cand_q[b].begin(), cl_cand_q[b].end());
        cl_checks[b].clear(); cl_qubits[b].clear(); cl_f_checks[b].clear(); cl_cand_q[b].clear();
        cluster_solved[a] = 0;
        return a;
    }

    void decode_bposd(const std::vector<int>& active_checks, const std::vector<uint8_t>& s_in,
                      std::vector<uint8_t>& corr_out) const {
        clear_corr(corr_out);
        if (active_checks.empty()) return;

        // 1. Extract 4-hop bipartite check-qubit active subgraph around `active_checks`
        //    (covers all paths of length <= 8 edges; equals full code automatically on small codes or dense defects):
        sub_c.clear(); sub_q.clear();
        std::vector<int> f_c = active_checks, f_q;
        for (int c : active_checks) { in_sub_c[c] = 1; sub_c.push_back(c); }
        for (int hop = 0; hop < 4; ++hop) {
            f_q.clear();
            for (int c : f_c) {
                for (int q : code.checks[c]) {
                    if (!in_sub_q[q]) { in_sub_q[q] = 1; sub_q.push_back(q); f_q.push_back(q); }
                }
            }
            if (hop == 3) break;
            f_c.clear();
            for (int q : f_q) {
                for (int c : code.q_to_checks[q]) {
                    if (!in_sub_c[c]) { in_sub_c[c] = 1; sub_c.push_back(c); f_c.push_back(c); }
                }
            }
            if (f_c.empty()) break;
        }

        // Initialize messages on active subgraph edges:
        const float L0 = 2.0f;
        const float alpha = 0.625f;
        const float damp = 0.50f;
        for (int c : sub_c) {
            for (int e_id = c_off[c]; e_id < c_off[c + 1]; ++e_id) {
                msg_q2c[e_id] = L0;
                msg_c2q[e_id] = 0.0f;
            }
        }
        std::vector<float> q_prior(sub_q.size(), L0);
        for (size_t qi = 0; qi < sub_q.size(); ++qi) {
            int q = sub_q[qi];
            int ext_c = 0;
            for (int c : code.q_to_checks[q]) if (!in_sub_c[c]) ++ext_c;
            q_prior[qi] = L0 + ext_c * (0.5f * L0);
            for (int k = q_off[q]; k < q_off[q + 1]; ++k) {
                msg_q2c[q_edges[k]] = q_prior[qi];
            }
        }

        bool bp_converged = false;
        for (int it = 0; it < 28; ++it) {
            // Check-to-variable update (Bounded Damped Normalized Min-Sum):
            for (int c : sub_c) {
                int e0 = c_off[c], e1 = c_off[c + 1];
                int parity_sgn = s_in[c] ? -1 : 1;
                float min1 = 1e30f, min2 = 1e30f;
                int min1_e = -1;
                for (int e_id = e0; e_id < e1; ++e_id) {
                    float v = msg_q2c[e_id];
                    if (v < 0.0f) parity_sgn = -parity_sgn;
                    float av = std::fabs(v);
                    if (av < min1) { min2 = min1; min1 = av; min1_e = e_id; }
                    else if (av < min2) { min2 = av; }
                }
                for (int e_id = e0; e_id < e1; ++e_id) {
                    float v = msg_q2c[e_id];
                    int sgn = (v < 0.0f) ? -parity_sgn : parity_sgn;
                    float mag = std::min(6.0f, (e_id == min1_e) ? min2 : min1);
                    float target = alpha * (float)sgn * mag;
                    msg_c2q[e_id] = (it == 0) ? target : ((1.0f - damp) * msg_c2q[e_id] + damp * target);
                }
            }

            // Variable-to-check & posterior update:
            for (size_t qi = 0; qi < sub_q.size(); ++qi) {
                int q = sub_q[qi];
                float tot = q_prior[qi];
                for (int k = q_off[q]; k < q_off[q + 1]; ++k) {
                    tot += msg_c2q[q_edges[k]];
                }
                post_L[q] = tot;
                for (int k = q_off[q]; k < q_off[q + 1]; ++k) {
                    int e_id = q_edges[k];
                    float m = tot - msg_c2q[e_id];
                    msg_q2c[e_id] = std::max(-12.0f, std::min(12.0f, m));
                }
            }

            bool ok = true;
            for (int c : sub_c) {
                uint8_t p = 0;
                for (int q : code.checks[c]) if (post_L[q] < 0.0f) p ^= 1;
                if (p != s_in[c]) { ok = false; break; }
            }
            if (ok) {
                for (int q : sub_q) {
                    if (post_L[q] < 0.0f) {
                        for (int c : code.q_to_checks[q]) {
                            if (!in_sub_c[c]) { ok = false; break; }
                        }
                        if (!ok) break;
                    }
                }
            }
            if (ok) {
                for (int q : sub_q) if (post_L[q] < 0.0f) flip_corr(q, corr_out);
                bp_converged = true;
                break;
            }
        }

        if (!bp_converged) {
            for (int q : sub_q) {
                soft_cost[q] = 1.0f + 0.15f * std::tanh(0.35f * post_L[q]);
            }

            if (code.N <= 1600) {
                std::vector<int> order(code.N);
                std::iota(order.begin(), order.end(), 0);
                for (int q = 0; q < code.N; ++q) {
                    if (!in_sub_q[q]) {
                        post_L[q] = L0 + (float)code.q_to_checks[q].size() * (0.5f * L0);
                        soft_cost[q] = 1.15f;
                    }
                }
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    if (post_L[a] != post_L[b]) return post_L[a] < post_L[b];
                    return code.q_to_checks[a].size() > code.q_to_checks[b].size();
                });
                std::vector<int> q_to_ord(code.N);
                std::vector<float> ord_cost(code.N);
                for (int j = 0; j < code.N; ++j) {
                    q_to_ord[order[j]] = j;
                    ord_cost[j] = soft_cost[order[j]];
                }
                std::vector<std::vector<int>> rcols(code.M_check);
                for (int ci = 0; ci < code.M_check; ++ci) {
                    rcols[ci].reserve(code.checks[ci].size());
                    for (int q : code.checks[ci]) rcols[ci].push_back(q_to_ord[q]);
                }
                std::vector<uint8_t> ord_sol;
                bool solved = solve_osd_cs_ordered(code.M_check, code.N, rcols, s_in, ord_cost, ord_sol, 1600);
                if (solved) {
                    for (int j = 0; j < code.N; ++j) if (ord_sol[j]) flip_corr(order[j], corr_out);
                }
            } else {
                // Pre-strip strictly syndrome-reducing qubits ordered by lowest BP posterior LLR
                // before running multi-scale BP-Ordered Cluster OSD-CS(36) on the residual traps:
                std::vector<uint8_t> res_syn = s_in;
                std::vector<int> cand_peel = sub_q;
                std::sort(cand_peel.begin(), cand_peel.end(), [&](int a, int b) {
                    return post_L[a] < post_L[b];
                });
                bool peeled_any = true;
                int peel_passes = 0;
                while (peeled_any && peel_passes < 6) {
                    peeled_any = false;
                    ++peel_passes;
                    for (int q : cand_peel) {
                        const auto& qc = code.q_to_checks[q];
                        int on = 0;
                        for (int c : qc) on += res_syn[c];
                        if (2 * on > (int)qc.size()) {
                            flip_corr(q, corr_out);
                            for (int c : qc) res_syn[c] ^= 1;
                            peeled_any = true;
                        }
                    }
                }
                std::vector<int> rem_act;
                for (int c = 0; c < code.M_check; ++c) if (res_syn[c]) rem_act.push_back(c);
                // Recompute fresh BP posteriors on the small residual trapping-set syndrome!
                if (!rem_act.empty()) {
                    for (int c : sub_c) {
                        for (int e_id = c_off[c]; e_id < c_off[c + 1]; ++e_id) {
                            msg_q2c[e_id] = L0;
                            msg_c2q[e_id] = 0.0f;
                        }
                    }
                    for (int it2 = 0; it2 < 16; ++it2) {
                        for (int c : sub_c) {
                            int e0 = c_off[c], e1 = c_off[c + 1];
                            int parity_sgn = res_syn[c] ? -1 : 1;
                            float min1 = 1e30f, min2 = 1e30f;
                            int min1_e = -1;
                            for (int e_id = e0; e_id < e1; ++e_id) {
                                float v = msg_q2c[e_id];
                                if (v < 0.0f) parity_sgn = -parity_sgn;
                                float av = std::fabs(v);
                                if (av < min1) { min2 = min1; min1 = av; min1_e = e_id; }
                                else if (av < min2) { min2 = av; }
                            }
                            for (int e_id = e0; e_id < e1; ++e_id) {
                                float v = msg_q2c[e_id];
                                int sgn = (v < 0.0f) ? -parity_sgn : parity_sgn;
                                float mag = std::min(6.0f, (e_id == min1_e) ? min2 : min1);
                                float target = alpha * (float)sgn * mag;
                                msg_c2q[e_id] = (it2 == 0) ? target : ((1.0f - damp) * msg_c2q[e_id] + damp * target);
                            }
                        }
                        for (size_t qi = 0; qi < sub_q.size(); ++qi) {
                            int q = sub_q[qi];
                            float tot = q_prior[qi];
                            for (int k = q_off[q]; k < q_off[q + 1]; ++k) tot += msg_c2q[q_edges[k]];
                            post_L[q] = tot;
                            soft_cost[q] = 1.0f + 0.15f * std::tanh(0.35f * tot);
                            for (int k = q_off[q]; k < q_off[q + 1]; ++k) {
                                int e_id = q_edges[k];
                                msg_q2c[e_id] = std::max(-12.0f, std::min(12.0f, tot - msg_c2q[e_id]));
                            }
                        }
                    }
                    run_compact_cluster_bposd(rem_act, res_syn, corr_out);
                }
            }
        }

        for (int c : sub_c) {
            in_sub_c[c] = 0;
            for (int e_id = c_off[c]; e_id < c_off[c + 1]; ++e_id) msg_c2q[e_id] = 0.0f;
        }
        for (int q : sub_q) {
            in_sub_q[q] = 0;
            post_L[q] = 4.0f;
            soft_cost[q] = 1.15f;
        }
    }

    void run_compact_cluster_bposd(const std::vector<int>& active_checks, const std::vector<uint8_t>& s_in,
                                   std::vector<uint8_t>& corr_out) const {
        // Multi-scale staged cluster radii (R = 24 -> 96 -> 384 -> 0) solved by BP-Ordered OSD-CS(36):
        std::vector<uint8_t> work_s(code.M_check, 0), in_work(code.M_check, 0);
        std::vector<int> touched_w, cur_act = active_checks;
        for (int c : active_checks) {
            work_s[c] = s_in[c];
            in_work[c] = 1;
            touched_w.push_back(c);
        }

        const int stage_caps[4] = {24, 96, 384, 0};
        for (int st = 0; st < 4 && !cur_act.empty(); ++st) {
            int max_loc_q = stage_caps[st];
            touched_nodes.clear();
            for (int c : cur_act) {
                res_s[c] = work_s[c];
                touch(c);
                parent[c] = c;
                cluster_solved[c] = 0;
                cl_checks[c].push_back(c);
                cl_f_checks[c].push_back(c);
            }
            std::vector<int> unsolved, cand_roots, stage_flips;
            while (true) {
                ++cur_stamp;
                unsolved.clear();
                for (int c : cur_act) {
                    if (!res_s[c]) continue;
                    int r = find_root(c);
                    if (!cluster_solved[r] && seen_stamp[r] != cur_stamp) {
                        seen_stamp[r] = cur_stamp;
                        unsolved.push_back(r);
                    }
                }
                if (unsolved.empty()) break;

                for (int r0 : unsolved) {
                    int r = find_root(r0);
                    std::vector<int> fc = std::move(cl_f_checks[r]);
                    cl_f_checks[r].clear();
                    for (int c : fc) {
                        for (int q : code.checks[c]) {
                            int qnode = code.M_check + q;
                            if (parent[qnode] == -1) {
                                touch(qnode);
                                parent[qnode] = r;
                                cl_qubits[r].push_back(q);
                                cl_cand_q[r].push_back(q);
                            } else {
                                r = merge_soft_clusters(r, qnode);
                            }
                            if (max_loc_q > 0 && (int)cl_qubits[r].size() > max_loc_q) cluster_solved[r] = 1;
                        }
                    }
                }

                ++cur_stamp;
                cand_roots.clear();
                for (int c : cur_act) {
                    if (!res_s[c]) continue;
                    int r = find_root(c);
                    if (!cluster_solved[r] && seen_stamp[r] != cur_stamp) {
                        seen_stamp[r] = cur_stamp;
                        cand_roots.push_back(r);
                    }
                }
                for (int r0 : cand_roots) {
                    int r = find_root(r0);
                    std::vector<int> fq = std::move(cl_cand_q[r]);
                    cl_cand_q[r].clear();
                    for (int q : fq) {
                        for (int c : code.q_to_checks[q]) {
                            if (parent[c] == -1) {
                                touch(c);
                                parent[c] = r;
                                res_s[c] = work_s[c];
                                cl_checks[r].push_back(c);
                                cl_f_checks[r].push_back(c);
                            } else {
                                r = merge_soft_clusters(r, c);
                            }
                            if (max_loc_q > 0 && (int)cl_qubits[r].size() > max_loc_q) cluster_solved[r] = 1;
                        }
                    }
                }

                ++cur_stamp;
                cand_roots.clear();
                for (int c : cur_act) {
                    if (!res_s[c]) continue;
                    int r = find_root(c);
                    if (!cluster_solved[r] && seen_stamp[r] != cur_stamp) {
                        seen_stamp[r] = cur_stamp;
                        cand_roots.push_back(r);
                    }
                }
                for (int r : cand_roots) {
                    auto& qs = cl_qubits[r];
                    const auto& cs = cl_checks[r];
                    if (qs.empty()) continue;
                    if (max_loc_q > 0 && (int)qs.size() > max_loc_q) {
                        cluster_solved[r] = 1;
                        continue;
                    }
                    int n_act_r = 0;
                    for (int ci : cs) if (res_s[ci]) ++n_act_r;
                    if (n_act_r == 0) { cluster_solved[r] = 1; continue; }

                    std::vector<int> ord_qs = qs;
                    std::sort(ord_qs.begin(), ord_qs.end(), [&](int a, int b) {
                        if (post_L[a] != post_L[b]) return post_L[a] < post_L[b];
                        return code.q_to_checks[a].size() > code.q_to_checks[b].size();
                    });

                    int nc = (int)ord_qs.size(), nr = (int)cs.size();
                    std::vector<float> loc_cost(nc);
                    for (int j = 0; j < nc; ++j) {
                        global_q_to_local[ord_qs[j]] = j;
                        loc_cost[j] = in_sub_q[ord_qs[j]] ? soft_cost[ord_qs[j]] : 1.15f;
                    }
                    std::vector<std::vector<int>> rcols(nr);
                    std::vector<uint8_t> rhs(nr, 0);
                    for (int i = 0; i < nr; ++i) {
                        int ci = cs[i];
                        rhs[i] = res_s[ci];
                        for (int q : code.checks[ci]) {
                            int loc = global_q_to_local[q];
                            if (loc != -1) rcols[i].push_back(loc);
                        }
                    }
                    for (int j = 0; j < nc; ++j) global_q_to_local[ord_qs[j]] = -1;

                    std::vector<uint8_t> lsol;
                    if (solve_osd_cs_ordered(nr, nc, rcols, rhs, loc_cost, lsol, 2400)) {
                        for (int j = 0; j < nc; ++j) {
                            if (lsol[j]) {
                                int q = ord_qs[j];
                                stage_flips.push_back(q);
                                for (int ci : code.q_to_checks[q]) res_s[ci] ^= 1;
                            }
                        }
                        cluster_solved[r] = 1;
                    }
                }
            }
            for (int node : touched_nodes) {
                if (node < code.M_check) res_s[node] = 0;
                parent[node] = -1;
                cluster_solved[node] = 0;
                cl_checks[node].clear();
                cl_qubits[node].clear();
                cl_f_checks[node].clear();
                cl_cand_q[node].clear();
            }
            for (int q : stage_flips) {
                flip_corr(q, corr_out);
                for (int c : code.q_to_checks[q]) {
                    if (!in_work[c]) { in_work[c] = 1; touched_w.push_back(c); }
                    work_s[c] ^= 1;
                }
            }
            cur_act.clear();
            for (int c : touched_w) if (work_s[c]) cur_act.push_back(c);
        }
    }
};

// =========================================================================================
// Independent Dual-Logical Finder (`prep_logicals`)
// =========================================================================================
void compute_and_append_dual_logicals(const std::string& raw_path, const std::string& out_path) {
    std::ifstream in(raw_path);
    if (!in) { std::cerr << "Cannot open raw " << raw_path << "\n"; std::exit(1); }
    int N, M_check, M_stab;
    in >> N >> M_check >> M_stab;
    std::vector<std::vector<int>> checks(M_check), stabs(M_stab);
    for (int i = 0; i < M_check; ++i) {
        int d; in >> d; checks[i].resize(d);
        for (int j = 0; j < d; ++j) in >> checks[i][j];
    }
    for (int i = 0; i < M_stab; ++i) {
        int d; in >> d; stabs[i].resize(d);
        for (int j = 0; j < d; ++j) in >> stabs[i][j];
    }
    std::vector<BitRow> stab_piv(N);
    std::vector<uint8_t> has_spiv(N, 0);
    std::vector<int> spiv_list;
    for (const auto& r : stabs) {
        BitRow x(N);
        for (int c : r) x.flip(c);
        while (true) {
            int p = x.first_one();
            if (p == -1) break;
            if (has_spiv[p]) x.xor_from(stab_piv[p], (size_t)(p >> 6));
            else { stab_piv[p] = std::move(x); has_spiv[p] = 1; spiv_list.push_back(p); break; }
        }
    }
    int rank_stab = (int)spiv_list.size();
    std::sort(spiv_list.rbegin(), spiv_list.rend());
    for (size_t i = 0; i < spiv_list.size(); ++i) {
        int p = spiv_list[i];
        size_t w0 = (size_t)(p >> 6);
        for (size_t j = i + 1; j < spiv_list.size(); ++j) {
            int p2 = spiv_list[j];
            if (stab_piv[p2].get(p)) stab_piv[p2].xor_from(stab_piv[p], w0);
        }
    }
    std::vector<BitRow> quot_piv(N);
    std::vector<uint8_t> has_qpiv(N, 0);
    int rank_check = 0;
    for (const auto& r : checks) {
        BitRow x(N);
        for (int c : r) x.flip(c);
        while (true) {
            int p = x.first_one();
            if (p == -1) break;
            if (has_qpiv[p]) x.xor_from(quot_piv[p], (size_t)(p >> 6));
            else { quot_piv[p] = std::move(x); has_qpiv[p] = 1; ++rank_check; break; }
        }
    }
    std::vector<std::vector<int>> dual_logs;
    for (int fc = 0; fc < N; ++fc) {
        if (has_spiv[fc]) continue;
        BitRow v(N); v.flip(fc);
        for (int p : spiv_list) if (stab_piv[p].get(fc)) v.flip(p);
        BitRow x = v;
        while (true) {
            int p = x.first_one();
            if (p == -1) break;
            if (has_qpiv[p]) x.xor_from(quot_piv[p], (size_t)(p >> 6));
            else {
                quot_piv[p] = std::move(x); has_qpiv[p] = 1;
                std::vector<int> supp;
                for (int q = 0; q < N; ++q) if (v.get(q)) supp.push_back(q);
                dual_logs.push_back(std::move(supp));
                break;
            }
        }
    }
    if ((int)dual_logs.size() != N - rank_check - rank_stab) {
        std::cerr << "FATAL K MISMATCH\n";
        std::exit(1);
    }
    std::ofstream out(out_path);
    out << N << " " << M_check << " " << M_stab << " " << dual_logs.size() << "\n";
    for (const auto& r : checks) { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
    for (const auto& r : stabs)  { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
    for (const auto& r : dual_logs) { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
}

// =========================================================================================
// Paired 4-Decoder Continuous-Time BKL Readout (`paired_readout`)
// Evaluates `uf`, `cool_uf`, `hier_uf`, and `bp_osd` on IDENTICAL thermal trajectories!
// =========================================================================================
void run_paired_readout_v19(
    const Code& code, double beta, int num_trials, int seed_offset,
    double log10_tmin, double log10_tmax, int num_grid,
    int64_t max_flips, int num_threads, bool verify_syn = false
) {
    int max_dE10 = 0;
    for (int q = 0; q < code.N; ++q) {
        int s = 0;
        for (int c : code.q_to_checks[q]) s += code.j_tenths[c];
        max_dE10 = std::max(max_dE10, s);
    }
    int nb_tot = 2 * max_dE10 + 1;
    std::vector<double> rates(nb_tot, 0.0);
    for (int b = 0; b < nb_tot; ++b) {
        double dE = 0.1 * (double)(b - max_dE10);
        rates[b] = 1.0 / (1.0 + std::exp(beta * dE));
    }

    std::vector<double> t_grid(num_grid, 0.0);
    for (int g = 0; g < num_grid; ++g) {
        double lg = (num_grid == 1) ? log10_tmin
                                    : (log10_tmin + (log10_tmax - log10_tmin) * (double)g / (double)(num_grid - 1));
        t_grid[g] = std::pow(10.0, lg);
    }

    std::vector<std::string> bits_uf(num_trials, std::string(num_grid, 'C'));
    std::vector<std::string> bits_cool(num_trials, std::string(num_grid, 'C'));
    std::vector<std::string> bits_hier(num_trials, std::string(num_grid, 'C'));
    std::vector<std::string> bits_bposd(num_trials, std::string(num_grid, 'C'));
    std::vector<int64_t> fl_counts(num_trials, 0);
    std::atomic<int> next_tr{0}, capped{0}, bad_syn{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int) {
        FastDecoderUF uf_dec(code), cool_dec(code), hier_dec(code), hier_stage(code);
        BPOSDDecoder bposd_dec(code);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_d(1.0);

        std::vector<uint8_t> e(code.N, 0), in_dirty_e(code.N, 0), s(code.M_check, 0);
        std::vector<uint8_t> c_uf(code.N, 0), c_cool(code.N, 0), c_hier(code.N, 0), c_stage(code.N, 0), c_bp(code.N, 0);
        std::vector<int> q_b(code.N, 0), q_p(code.N, 0), act, pos_act(code.M_check, -1), dirty_e;
        std::vector<std::vector<int>> buckets(nb_tot);
        std::vector<double> Q_cum(nb_tot, 0.0);

        auto set_b = [&](int q, int new_b) {
            int old_b = q_b[q];
            if (old_b == new_b) return;
            int p = q_p[q], lq = buckets[old_b].back();
            buckets[old_b][p] = lq; q_p[lq] = p; buckets[old_b].pop_back();
            q_b[q] = new_b; q_p[q] = (int)buckets[new_b].size(); buckets[new_b].push_back(q);
        };
        auto tog_c = [&](int c) {
            if (!s[c]) { s[c] = 1; pos_act[c] = (int)act.size(); act.push_back(c); }
            else {
                s[c] = 0; int p = pos_act[c], lc = act.back();
                act[p] = lc; pos_act[lc] = p; act.pop_back(); pos_act[c] = -1;
            }
        };

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            int g_tr = seed_offset + tr;
            // Exact same RNG seed formula as kmc_engine_v17!
            std::mt19937_64 rng(887766551ULL + 104729ULL * (uint64_t)(g_tr + 1)
                                + (uint64_t)(beta * 1000) + (uint64_t)code.N * 97ULL
                                + (uint64_t)code.K * 1000003ULL + (uint64_t)code.M_stab * 37ULL);

            for (int q : dirty_e) { e[q] = 0; in_dirty_e[q] = 0; } dirty_e.clear();
            for (int c : act) { s[c] = 0; pos_act[c] = -1; } act.clear();
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < code.N; ++q) {
                int dE10 = 0;
                for (int c : code.q_to_checks[q]) dE10 += code.j_tenths[c];
                int rb = dE10 + max_dE10;
                q_b[q] = rb; q_p[q] = (int)buckets[rb].size(); buckets[rb].push_back(q);
            }

            double t = 0.0;
            int64_t flips = 0;
            int g = 0;
            bool cr_uf = false, cr_cool = false, cr_hier = false, cr_bp = false;
            std::string r_uf(num_grid, 'C'), r_cool(num_grid, 'C'), r_hier(num_grid, 'C'), r_bp(num_grid, 'C');

            while (g < num_grid && flips < max_flips) {
                double Qt = 0.0;
                for (int b = 0; b < nb_tot; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                double dt = exp_d(rng) / Qt;
                double t_new = t + dt;

                if (g < num_grid && t_grid[g] <= t_new) {
                    bool e_uf = false, e_cool = false, e_hier = false, e_bp = false;
                    bool is_zero_syn = act.empty();
                    bool cooled_zero = true;
                    if (is_zero_syn) {
                        bool raw = code.has_logical_error_raw(e);
                        e_uf = e_cool = e_hier = e_bp = raw;
                    } else {
                        uf_dec.decode_uf(act, s, c_uf);
                        e_uf = code.has_logical_error(e, c_uf);

                        cool_dec.decode_cool_uf(act, c_cool, cooled_zero);
                        e_cool = code.has_logical_error(e, c_cool);

                        hier_dec.decode_hier_uf(act, hier_stage, c_stage, c_hier);
                        e_hier = code.has_logical_error(e, c_hier);

                        bposd_dec.decode_bposd(act, s, c_bp);
                        e_bp = code.has_logical_error(e, c_bp);

                        if (verify_syn) {
                            if (!code.check_syndrome_match(s, c_uf))   { std::cerr << "BAD_UF\n"; bad_syn.fetch_add(1); }
                            if (!code.check_syndrome_match(s, c_cool)) { std::cerr << "BAD_COOL\n"; bad_syn.fetch_add(1); }
                            if (!code.check_syndrome_match(s, c_hier)) { std::cerr << "BAD_HIER\n"; bad_syn.fetch_add(1); }
                            if (!code.check_syndrome_match(s, c_bp))   { std::cerr << "BAD_BPOSD\n"; bad_syn.fetch_add(1); }
                        }
                    }
                    if (e_uf) cr_uf = true;
                    if (e_cool) cr_cool = true;
                    if (e_hier) cr_hier = true;
                    if (e_bp) cr_bp = true;

                    // Failure basin tags when a decoder fails:
                    //   'G' = exact zero-syndrome wrong ground state (s == 0, E == 0: unrecoverable by ANY decoder!)
                    //   'Q' = strictly downhill T=0 cooling reaches s_cool == 0 in wrong logical basin!
                    //   '0' = failed with excited residual syndrome
                    //   '1' = recovered (had prior grid failure)
                    //   '2' = recovered (no prior grid failure)
                    auto fail_tag = [&](bool err) -> char {
                        if (!err) return '0'; // unused
                        if (is_zero_syn) return 'G';
                        if (cooled_zero && e_cool) return 'Q';
                        return '0';
                    };
                    char ch_uf   = e_uf   ? fail_tag(true) : (cr_uf   ? '1' : '2');
                    char ch_cool = e_cool ? fail_tag(true) : (cr_cool ? '1' : '2');
                    char ch_hier = e_hier ? fail_tag(true) : (cr_hier ? '1' : '2');
                    char ch_bp   = e_bp   ? fail_tag(true) : (cr_bp   ? '1' : '2');

                    while (g < num_grid && t_grid[g] <= t_new) {
                        r_uf[g] = ch_uf;
                        r_cool[g] = ch_cool;
                        r_hier[g] = ch_hier;
                        r_bp[g] = ch_bp;
                        ++g;
                    }
                    if (g >= num_grid) {
                        t = t_new;
                        break;
                    }
                }

                t = t_new;
                double r = uni(rng) * Qt;
                int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                if (cb >= nb_tot) cb = nb_tot - 1;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < nb_tot - 1) ++cb;

                int q = buckets[cb][rng() % buckets[cb].size()];
                if (!in_dirty_e[q]) { in_dirty_e[q] = 1; dirty_e.push_back(q); }
                e[q] ^= 1; ++flips;
                for (int c : code.q_to_checks[q]) {
                    uint8_t was = s[c];
                    tog_c(c);
                    int db = (was ? +2 : -2) * code.j_tenths[c];
                    for (int q2 : code.checks[c]) set_b(q2, q_b[q2] + db);
                }
            }

            if (g < num_grid) capped.fetch_add(1);
            bits_uf[tr] = std::move(r_uf);
            bits_cool[tr] = std::move(r_cool);
            bits_hier[tr] = std::move(r_hier);
            bits_bposd[tr] = std::move(r_bp);
            fl_counts[tr] = flips;
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    if (bad_syn.load() > 0) {
        std::cerr << "FATAL SYNDROME MISMATCH IN PAIRED DECODERS: bad_syn=" << bad_syn.load() << "\n";
        std::exit(1);
    }

    double mf = 0.0;
    for (int i = 0; i < num_trials; ++i) mf += (double)fl_counts[i];
    mf /= num_trials;

    auto print_bits = [](const std::vector<std::string>& v) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << v[i];
        }
    };

    std::cout << "PAIRED_READOUT beta=" << std::fixed << std::setprecision(5) << beta
              << " N=" << code.N << " K=" << code.K
              << " trials=" << num_trials << " seed_off=" << seed_offset
              << " grid=" << std::fixed << std::setprecision(4) << log10_tmin << ":" << log10_tmax << ":" << num_grid
              << " mean_flips=" << std::fixed << std::setprecision(1) << mf
              << " capped=" << capped.load()
              << " bits_uf=";
    print_bits(bits_uf);
    std::cout << " bits_cool=";
    print_bits(bits_cool);
    std::cout << " bits_hier=";
    print_bits(bits_hier);
    std::cout << " bits_bposd=";
    print_bits(bits_bposd);
    std::cout << "\n";
}

// =========================================================================================
// Paired 4-Decoder Unrelaxed i.i.d. Bit-Flip Benchmark (`iid_readout`)
// Evaluates `uf`, `cool_uf`, `hier_uf`, and `bp_osd` on IDENTICAL i.i.d. error realizations across p!
// =========================================================================================
void run_iid_readout_v19(
    const Code& code, int num_trials, int seed_offset,
    double p_min, double p_max, int num_grid, int num_threads
) {
    std::vector<double> p_grid(num_grid, 0.0);
    for (int g = 0; g < num_grid; ++g) {
        p_grid[g] = (num_grid == 1) ? p_min : (p_min + (p_max - p_min) * (double)g / (double)(num_grid - 1));
    }
    std::vector<std::string> bits_uf(num_trials, std::string(num_grid, '0'));
    std::vector<std::string> bits_cool(num_trials, std::string(num_grid, '0'));
    std::vector<std::string> bits_hier(num_trials, std::string(num_grid, '0'));
    std::vector<std::string> bits_bposd(num_trials, std::string(num_grid, '0'));
    std::atomic<int> next_tr{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int) {
        FastDecoderUF uf_dec(code), cool_dec(code), hier_dec(code), hier_stage(code);
        BPOSDDecoder bposd_dec(code);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<double> u_q(code.N, 0.0);
        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0);
        std::vector<uint8_t> c_uf(code.N, 0), c_cool(code.N, 0), c_hier(code.N, 0), c_stage(code.N, 0), c_bp(code.N, 0);
        std::vector<int> act;

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            int g_tr = seed_offset + tr;
            std::mt19937_64 rng(445566771ULL + 104729ULL * (uint64_t)(g_tr + 1) + (uint64_t)code.N * 131ULL);
            // Coupled monotone uniform draws u_q in [0,1) per qubit across p_grid:
            for (int q = 0; q < code.N; ++q) u_q[q] = uni(rng);

            for (int g = 0; g < num_grid; ++g) {
                double p = p_grid[g];
                std::fill(s.begin(), s.end(), 0);
                act.clear();
                for (int q = 0; q < code.N; ++q) {
                    uint8_t eq = (u_q[q] < p) ? 1 : 0;
                    e[q] = eq;
                    if (eq) {
                        for (int c : code.q_to_checks[q]) s[c] ^= 1;
                    }
                }
                for (int c = 0; c < code.M_check; ++c) if (s[c]) act.push_back(c);

                bool e_uf = false, e_cool = false, e_hier = false, e_bp = false;
                bool cooled_zero = true;
                if (act.empty()) {
                    e_uf = e_cool = e_hier = e_bp = code.has_logical_error_raw(e);
                } else {
                    uf_dec.decode_uf(act, s, c_uf);
                    e_uf = code.has_logical_error(e, c_uf);

                    cool_dec.decode_cool_uf(act, c_cool, cooled_zero);
                    e_cool = code.has_logical_error(e, c_cool);

                    hier_dec.decode_hier_uf(act, hier_stage, c_stage, c_hier);
                    e_hier = code.has_logical_error(e, c_hier);

                    bposd_dec.decode_bposd(act, s, c_bp);
                    e_bp = code.has_logical_error(e, c_bp);
                }
                bits_uf[tr][g]    = e_uf   ? '0' : '2';
                bits_cool[tr][g]  = e_cool ? '0' : '2';
                bits_hier[tr][g]  = e_hier ? '0' : '2';
                bits_bposd[tr][g] = e_bp   ? '0' : '2';
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    auto print_bits = [](const std::vector<std::string>& v) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << v[i];
        }
    };
    std::cout << "IID_READOUT N=" << code.N << " K=" << code.K
              << " trials=" << num_trials << " seed_off=" << seed_offset
              << " pgrid=" << std::fixed << std::setprecision(5) << p_min << ":" << p_max << ":" << num_grid
              << " bits_uf=";
    print_bits(bits_uf);
    std::cout << " bits_cool=";
    print_bits(bits_cool);
    std::cout << " bits_hier=";
    print_bits(bits_hier);
    std::cout << " bits_bposd=";
    print_bits(bits_bposd);
    std::cout << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    std::string mode = argv[1];
    int hw = std::max(1, (int)std::thread::hardware_concurrency());
    if (mode == "verify") {
        Code c = Code::load(argv[2]);
        int bad = c.verify_commutation();
        std::cout << "VERIFY " << argv[2] << " N=" << c.N << " M_check=" << c.M_check
                  << " M_stab=" << c.M_stab << " K=" << c.K << " bad_comm=" << bad << "\n";
        return bad == 0 ? 0 : 1;
    }
    if (mode == "prep_logicals") {
        compute_and_append_dual_logicals(argv[2], argv[3]);
        return 0;
    }
    if (mode == "paired_readout" || mode == "self_test") {
        Code c = Code::load(argv[2]);
        bool vsyn = (mode == "self_test");
        run_paired_readout_v19(
            c, std::stod(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]),
            std::stod(argv[6]), std::stod(argv[7]), std::stoi(argv[8]),
            std::stoll(argv[9]), (argc >= 11) ? std::stoi(argv[10]) : hw, vsyn
        );
        return 0;
    }
    if (mode == "iid_readout") {
        Code c = Code::load(argv[2]);
        run_iid_readout_v19(
            c, std::stoi(argv[3]), std::stoi(argv[4]),
            std::stod(argv[5]), std::stod(argv[6]), std::stoi(argv[7]),
            (argc >= 9) ? std::stoi(argv[8]) : hw
        );
        return 0;
    }
    return 1;
}
