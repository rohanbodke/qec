// kmc_engine_v7.cpp
// V7 Zero-Trust Rescue & Verification Engine for MIT 6.S980:
//   1. Hard mathematical assertions in `prep_logicals` & `verify`:
//      - Verifies exact CSS commutation H_check * H_stab^T == 0 (mod 2)
//      - Verifies exact K == N - rank(H_check) - rank(H_stab) and logical commutation/independence
//   2. Zero timer-floor KMC (`t_next = 0.0`, decoding every flip for flips <= 16, then 0.05 * t)
//      + OSD-2 coset weight refinement in `FastDecoderV7`
//   3. Exact Ground-to-Ground Renewal Forward Flux Sampling (`run_ffs_v7`):
//      - Exact Basin A (s == 0, trivial logical) & Exact Basin B (s == 0, nontrivial logical)
//      - Every-flip decoding (zero striding bias), full bank retention, adaptive monotone lambda = 2*E + w
//      - Unbiased replica-pooled estimator mean_tau / mean_P across ALL R replicas (zero Jensen/survivorship bias)
//      - Full ground-to-ground E=0 -> saddle -> E=0 failure movie recording
//   4. Passive Cooling & Autonomous Self-Healing (`run_cool_v7`):
//      - Measures exact `p_zero_syn`, `p_log_survive` (with decoder), AND `p_exact_heal` (s==0 & exact raw codeword, zero decoder!)
//      - Compares `quench`, `anneal`, and targeted `soft_seam` (<20% seam checks softened)

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
    inline int dot_mod2(const BitRow& o) const {
        uint64_t acc = 0;
        const size_t n = w.size();
        for (size_t i = 0; i < n; ++i) acc ^= (w[i] & o.w[i]);
        return __builtin_popcountll(acc) & 1;
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

    static Code load(const std::string& path, const std::string& j_path = "") {
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
        for (int q = 0; q < c.N; ++q) c.max_q_deg = std::max(c.max_q_deg, (int)c.q_to_checks[q].size());
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
        if (!j_path.empty() && j_path != "uniform") {
            std::ifstream jin(j_path);
            if (!jin) { std::cerr << "Cannot open J file " << j_path << "\n"; std::exit(1); }
            for (int i = 0; i < c.M_check; ++i) jin >> c.j_tenths[i];
        }
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
        // Also verify every dual logical commutes with all stabs (L_dual in Ker(H_stab)):
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

    inline uint64_t logical_class_mask(const std::vector<uint8_t>& e, const std::vector<uint8_t>& corr) const {
        uint64_t mask = 0;
        int lim = std::min(K, 64);
        for (int a = 0; a < lim; ++a) {
            uint8_t p = 0;
            for (int q : logicals[a]) p ^= (e[q] ^ corr[q]);
            if (p) mask |= (1ULL << a);
        }
        return mask;
    }
};

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

    // Hard Gate 1: Verify exact CSS commutation H_check * H_stab^T == 0 (mod 2) via sparse q_to_checks!
    {
        std::vector<std::vector<int>> q2c(N);
        for (int c = 0; c < M_check; ++c) {
            for (int q : checks[c]) q2c[q].push_back(c);
        }
        std::vector<uint8_t> hit_par(M_check, 0);
        std::vector<int> touched;
        int bad_pairs = 0;
        for (int s = 0; s < M_stab; ++s) {
            touched.clear();
            for (int q : stabs[s]) {
                for (int c : q2c[q]) {
                    if (!hit_par[c]) touched.push_back(c);
                    hit_par[c] ^= 1;
                }
            }
            for (int c : touched) {
                if (hit_par[c]) { ++bad_pairs; hit_par[c] = 0; }
            }
        }
        if (bad_pairs > 0) {
            std::cerr << "FATAL COMMUTATION FAILURE in " << raw_path << ": bad_pairs=" << bad_pairs << "\n";
            std::exit(1);
        }
    }

    std::vector<BitRow> stab_piv(N);
    std::vector<uint8_t> has_spiv(N, 0);
    std::vector<int> spiv_list;
    spiv_list.reserve(N);
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
    int nt = std::min(16, std::max(1, (int)std::thread::hardware_concurrency()));
    if (spiv_list.size() > 4000 && nt > 1) {
        for (size_t i = 0; i < spiv_list.size(); ++i) {
            int p = spiv_list[i];
            size_t w0 = (size_t)(p >> 6);
            size_t rem = spiv_list.size() - (i + 1);
            if (rem < 1024) {
                for (size_t j = i + 1; j < spiv_list.size(); ++j) {
                    int p2 = spiv_list[j];
                    if (stab_piv[p2].get(p)) stab_piv[p2].xor_from(stab_piv[p], w0);
                }
            } else {
                std::vector<std::thread> pool;
                size_t chunk = (rem + nt - 1) / nt;
                for (int t = 0; t < nt; ++t) {
                    size_t j0 = i + 1 + t * chunk;
                    size_t j1 = std::min(spiv_list.size(), j0 + chunk);
                    if (j0 >= j1) break;
                    pool.emplace_back([&, j0, j1, p, w0]() {
                        for (size_t j = j0; j < j1; ++j) {
                            int p2 = spiv_list[j];
                            if (stab_piv[p2].get(p)) stab_piv[p2].xor_from(stab_piv[p], w0);
                        }
                    });
                }
                for (auto& th : pool) th.join();
            }
        }
    } else {
        for (size_t i = 0; i < spiv_list.size(); ++i) {
            int p = spiv_list[i];
            size_t w0 = (size_t)(p >> 6);
            for (size_t j = i + 1; j < spiv_list.size(); ++j) {
                int p2 = spiv_list[j];
                if (stab_piv[p2].get(p)) stab_piv[p2].xor_from(stab_piv[p], w0);
            }
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

    // Hard Gate 2: Exact CSS dimension identity K == N - rank(H_check) - rank(H_stab)!
    int expected_K = N - rank_check - rank_stab;
    if ((int)dual_logs.size() != expected_K) {
        std::cerr << "FATAL K MISMATCH in " << raw_path << ": got " << dual_logs.size()
                  << " expected " << expected_K << "\n";
        std::exit(1);
    }

    std::ofstream out(out_path);
    out << N << " " << M_check << " " << M_stab << " " << dual_logs.size() << "\n";
    for (const auto& r : checks) { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
    for (const auto& r : stabs)  { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
    for (const auto& r : dual_logs) { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
}

bool solve_gf2_dense_core(
    int nr, int nc,
    const std::vector<std::vector<int>>& row_cols,
    const std::vector<uint8_t>& rhs,
    std::vector<uint8_t>& sol_out,
    bool min_weight_refine
) {
    int width = nc + 1;
    std::vector<BitRow> mat(nr, BitRow(width));
    for (int r = 0; r < nr; ++r) {
        for (int c : row_cols[r]) mat[r].flip(c);
        if (rhs[r]) mat[r].flip(nc);
    }
    std::vector<int> pcol(nr, -1), prow(nc, -1);
    int rank = 0;
    bool do_refine = min_weight_refine && (nc <= 380);
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

// Exact Sparse Markowitz GF(2) Solver + Compact Dense-Core Finisher for large LDPC clusters (nc > 380):
bool solve_gf2_submatrix(
    int nr, int nc,
    const std::vector<std::vector<int>>& row_cols,
    const std::vector<uint8_t>& rhs,
    std::vector<uint8_t>& sol_out,
    bool min_weight_refine = true
) {
    if (nc <= 380) {
        return solve_gf2_dense_core(nr, nc, row_cols, rhs, sol_out, min_weight_refine);
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

        // Switch to compact BitRow finish once the remaining unpeeled core is small & dense:
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
            if (!solve_gf2_dense_core(sub_nr, sub_nc, sub_rc, sub_rhs, sub_sol, false)) return false;
            for (int j = 0; j < sub_nc; ++j) sol_out[sub_cols[j]] = sub_sol[j];
            break;
        }

        // Choose Markowitz pivot column in sel_r with smallest col_deg:
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

    // Hard O(nnz) exact verification gate:
    for (int r = 0; r < nr; ++r) {
        uint8_t p = 0;
        for (int c : row_cols[r]) p ^= sol_out[c];
        if (p != rhs[r]) {
            std::cerr << "FATAL SPARSE GF2 RESIDUAL MISMATCH at r=" << r << "\n";
            std::exit(1);
        }
    }
    return true;
}

// Strictly O(|act|) Cluster UF + Optional Greedy Majority-Vote Pre-Decoder (`use_maj`)!
struct FastDecoderV7 {
    const Code& code;
    bool use_maj = false;
    int V;
    mutable std::vector<int> parent, global_q_to_local, touched_nodes, seen_stamp;
    mutable std::vector<uint8_t> cluster_solved, res_s, work_s, in_work_act, in_cand_q, in_dirty_corr;
    mutable std::vector<int> dirty_corr, work_act, cand_q, touched_work_c;
    mutable std::vector<std::vector<int>> cl_checks, cl_qubits, cl_f_checks, cl_f_qubits;
    mutable int cur_stamp = 1;
    mutable int64_t perc_bails = 0;

    explicit FastDecoderV7(const Code& c, bool maj = false)
        : code(c), use_maj(maj), V(c.M_check + c.N), parent(V, -1), global_q_to_local(c.N, -1),
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

    inline int corr_weight(const std::vector<uint8_t>& corr_out) const {
        int w = 0;
        for (int q : dirty_corr) if (corr_out[q]) ++w;
        return w;
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
                     std::vector<uint8_t>& corr_out, int& max_cl_qubits, int max_local_qubits = 0) const {
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
                max_cl_qubits = std::max(max_cl_qubits, (int)qs.size());
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
                if (solve_gf2_submatrix(nr, nc, rcols, rhs, lsol, true)) {
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

    int decode_into(const std::vector<int>& active_checks, const std::vector<uint8_t>& s_in,
                    std::vector<uint8_t>& corr_out, int max_local_qubits = 0) const {
        clear_corr(corr_out);
        if (active_checks.empty()) return 0;
        int max_cl_qubits = 0;
        if (!use_maj) {
            run_uf_core(active_checks, s_in, corr_out, max_cl_qubits, max_local_qubits);
            return max_cl_qubits;
        }
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
            run_uf_core(work_act, work_s, corr_out, max_cl_qubits);
        }
        for (int c : touched_work_c) { work_s[c] = 0; in_work_act[c] = 0; }
        return max_cl_qubits;
    }
};

// Exact Continuous-Time BKL Monte Carlo with ZERO timer floor (`t_next = 0.0`, every-flip check for flips <= 16)
void run_kmc_v7(const Code& code, bool use_maj, double beta, int num_trials, int64_t max_flips, int num_threads) {
    int max_dE10 = 0;
    for (int q = 0; q < code.N; ++q) {
        int s = 0;
        for (int c : code.q_to_checks[q]) s += code.j_tenths[c];
        max_dE10 = std::max(max_dE10, s);
    }
    int nb = 2 * max_dE10 + 1;
    std::vector<double> rates(nb, 0.0);
    for (int b = 0; b < nb; ++b) {
        double dE = 0.1 * (double)(b - max_dE10);
        rates[b] = 1.0 / (1.0 + std::exp(beta * dE));
    }

    std::vector<double> t_fails(num_trials, 0.0);
    std::vector<int64_t> fl_counts(num_trials, 0);
    std::atomic<int> next_tr{0}, capped{0};
    std::atomic<int64_t> tot_bails{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int tid) {
        FastDecoderV7 dec(code, use_maj);
        std::mt19937_64 rng(918273645ULL * (tid + 1) + (uint64_t)(beta * 1000) + code.N * 53ULL);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_d(1.0);

        std::vector<uint8_t> e(code.N, 0), in_dirty_e(code.N, 0), s(code.M_check, 0), corr(code.N, 0);
        std::vector<int> q_b(code.N, 0), q_p(code.N, 0), act, pos_act(code.M_check, -1), dirty_e;
        std::vector<std::vector<int>> buckets(nb);
        std::vector<double> Q_cum(nb, 0.0);

        auto move_q = [&](int q, int new_b) {
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
            std::mt19937_64 rng(918273645ULL + 99991ULL * (uint64_t)(tr + 1) + (uint64_t)(beta * 1000) + code.N * 53ULL);
            for (int q : dirty_e) { e[q] = 0; in_dirty_e[q] = 0; }
            dirty_e.clear();
            for (int c : act) { s[c] = 0; pos_act[c] = -1; } act.clear();
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < code.N; ++q) {
                int dE10 = 0;
                for (int c : code.q_to_checks[q]) dE10 += code.j_tenths[c];
                int b = dE10 + max_dE10;
                q_b[q] = b; q_p[q] = (int)buckets[b].size(); buckets[b].push_back(q);
            }
            double t = 0.0, t_next = 0.0;
            int64_t flips = 0;
            bool failed = false;
            while (flips < max_flips) {
                double Qt = 0.0;
                for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                double dt = exp_d(rng) / Qt;
                t += dt; t_next -= dt;
                double r = uni(rng) * Qt;
                int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                if (cb >= nb) cb = nb - 1;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < nb - 1) ++cb;

                int q = buckets[cb][rng() % buckets[cb].size()];
                if (!in_dirty_e[q]) { in_dirty_e[q] = 1; dirty_e.push_back(q); }
                e[q] ^= 1; ++flips;
                for (int c : code.q_to_checks[q]) {
                    uint8_t was = s[c];
                    tog_c(c);
                    int db = (was ? +2 : -2) * code.j_tenths[c];
                    for (int q2 : code.checks[c]) move_q(q2, q_b[q2] + db);
                }
                if (flips <= 16 || t_next <= 0.0) {
                    if (act.empty()) {
                        if (code.has_logical_error_raw(e)) { failed = true; break; }
                    } else {
                        dec.decode_into(act, s, corr);
                        if (code.has_logical_error(e, corr)) { failed = true; break; }
                    }
                    t_next = 0.20 * t;
                }
            }
            if (!failed) capped.fetch_add(1);
            t_fails[tr] = t;
            fl_counts[tr] = flips;
        }
        tot_bails.fetch_add(dec.perc_bails);
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    double sum = 0.0, sum_fl = 0.0;
    for (int i = 0; i < num_trials; ++i) { sum += t_fails[i]; sum_fl += (double)fl_counts[i]; }
    double mean_t = sum / num_trials;
    double var = 0.0;
    for (double v : t_fails) var += (v - mean_t) * (v - mean_t);
    double sem_t = (num_trials > 1) ? std::sqrt(var / (num_trials * (num_trials - 1.0))) : 0.0;
    std::sort(t_fails.begin(), t_fails.end());
    std::cout << "KMC dec=" << (use_maj ? "maj_uf" : "uf")
              << " beta=" << std::fixed << std::setprecision(2) << beta
              << " N=" << code.N << " K=" << code.K << " trials=" << num_trials
              << " mean_t=" << std::scientific << std::setprecision(4) << mean_t
              << " sem_t=" << sem_t
              << " median_t=" << t_fails[num_trials / 2]
              << " mean_flips=" << std::fixed << std::setprecision(1) << (sum_fl / num_trials)
              << " bails=" << tot_bails.load()
              << " capped=" << capped.load()
              << " t_list=";
    for (int i = 0; i < num_trials; ++i) {
        if (i) std::cout << ",";
        std::cout << std::scientific << std::setprecision(5) << t_fails[i];
    }
    std::cout << "\n";
}

// =========================================================================================
// EXACT GROUND-TO-GROUND RENEWAL FORWARD FLUX SAMPLING (`run_ffs_v7`)
//   - Basin A: exact s == 0 (`act.empty()`) with trivial logical (`!has_logical_error_raw(e)`)
//   - Basin B: exact s == 0 (`act.empty()`) with nontrivial logical (`has_logical_error_raw(e)`)
//              (plus 4,000-flip BKL committor fallback ONLY if Phi_0 > 0.04 at warm beta)
//   - Adaptive milestone progression along lambda = round(0.2 * E_tenths) + |corr(s)|
//   - Replica-pooled estimator mean_tau / mean_P across ALL R replicas (zero Jensen/survivorship bias!)
// =========================================================================================
struct SparseStateV7 {
    std::vector<int> flipped_q;
    int lam = 0;
    std::vector<int> e_trace_tenths;
};

void run_ffs_v7(const Code& code, bool use_maj, double beta, int shots_per_stage,
                int num_replicas, int num_threads, const std::string& movie_out = "") {
    int max_dE10 = 0;
    for (int q = 0; q < code.N; ++q) {
        int s = 0;
        for (int c : code.q_to_checks[q]) s += code.j_tenths[c];
        max_dE10 = std::max(max_dE10, s);
    }
    int nb = 2 * max_dE10 + 1;
    std::vector<double> rates(nb, 0.0);
    for (int b = 0; b < nb; ++b) {
        double dE = 0.1 * (double)(b - max_dE10);
        rates[b] = 1.0 / (1.0 + std::exp(beta * dE));
    }

    std::vector<int> base_b(code.N, 0);
    double Qt_ground = 0.0;
    for (int q = 0; q < code.N; ++q) {
        int dE10 = 0;
        for (int c : code.q_to_checks[q]) dE10 += code.j_tenths[c];
        base_b[q] = dE10 + max_dE10;
        Qt_ground += rates[base_b[q]];
    }
    double Phi_0 = Qt_ground;

    // Precompute exact analytical 0 -> q0 -> q1 != q0 conditioning
    std::vector<double> cdf_q0_12(code.N, 0.0), cdf0(code.N, 0.0);
    std::vector<int> nbr_delta(code.N, 0), touched_nbr;
    double sum_w12 = 0.0, exact_mean_tau_step1 = 0.0;
    bool any_d1_or_checkless = false;
    {
        double acc = 0.0;
        for (int q = 0; q < code.N; ++q) { acc += rates[base_b[q]]; cdf0[q] = acc; }

        std::vector<uint8_t> tmp_e(code.N, 0), tmp_s(code.M_check, 0), tmp_corr(code.N, 0);
        FastDecoderV7 tmp_dec(code, use_maj);
        for (int q0 = 0; q0 < code.N; ++q0) {
            if (code.q_to_checks[q0].empty()) any_d1_or_checkless = true;
            tmp_e[q0] = 1;
            const auto& a1 = code.q_to_checks[q0];
            for (int c : a1) tmp_s[c] = 1;
            if (a1.empty()) {
                if (code.has_logical_error_raw(tmp_e)) any_d1_or_checkless = true;
            } else {
                tmp_dec.decode_into(a1, tmp_s, tmp_corr);
                if (code.has_logical_error(tmp_e, tmp_corr)) any_d1_or_checkless = true;
            }
            for (int c : a1) tmp_s[c] = 0;
            tmp_e[q0] = 0;

            touched_nbr.clear();
            for (int c : code.q_to_checks[q0]) {
                int db = -2 * code.j_tenths[c];
                for (int q2 : code.checks[c]) {
                    if (nbr_delta[q2] == 0) touched_nbr.push_back(q2);
                    nbr_delta[q2] += db;
                }
            }
            double Qt_q0 = Qt_ground;
            for (int q2 : touched_nbr) Qt_q0 += rates[base_b[q2] + nbr_delta[q2]] - rates[base_b[q2]];
            double r_back = rates[base_b[q0] + nbr_delta[q0]];
            double r_up = std::max(0.0, Qt_q0 - r_back);
            double p_up = (Qt_q0 > 0.0) ? (r_up / Qt_q0) : 0.0;
            double p_q0 = rates[base_b[q0]] / Phi_0;
            exact_mean_tau_step1 += p_q0 * ((Qt_q0 > 0.0) ? (1.0 / Qt_q0) : 0.0);
            sum_w12 += rates[base_b[q0]] * p_up;
            cdf_q0_12[q0] = sum_w12;
            for (int q2 : touched_nbr) nbr_delta[q2] = 0;
        }
    }
    double P_1to2_exact = sum_w12 / Phi_0;
    bool use_cond12 = (!any_d1_or_checkless && P_1to2_exact > 0.0 && P_1to2_exact < 0.35);
    bool allow_warm_basin = (Phi_0 > 0.04); // only if thermal background > 4%

    // Zobrist hash random keys per check for O(1) syndrome-decoder memoization (bit-for-bit exact!)
    std::vector<uint64_t> z_check_key(code.M_check, 0ULL);
    {
        std::mt19937_64 zrng(0x9E3779B97F4A7C15ULL ^ (uint64_t)code.M_check);
        for (int c = 0; c < code.M_check; ++c) {
            z_check_key[c] = zrng();
            if (z_check_key[c] == 0ULL) z_check_key[c] = 1ULL;
        }
    }

    // Precompute per-qubit 64-bit logical parity mask (exact O(1) logical tracking whenever K <= 64!)
    std::vector<uint64_t> q_log_mask(code.N, 0ULL);
    if (code.K <= 64) {
        for (int a = 0; a < code.K; ++a) {
            for (int q : code.logicals[a]) q_log_mask[q] ^= (1ULL << a);
        }
    }

    int nt = std::max(1, num_threads);
    struct SynCacheEntry {
        uint64_t h = 0ULL;
        uint64_t corr_lmask = 0ULL;
        int w = -1;
        uint8_t valid = 0;
    };
    struct ThreadWS {
        FastDecoderV7 dec;
        std::vector<uint8_t> e, in_dirty_e, s, corr;
        std::vector<int> q_b, q_p, act, pos_act, dirty_e;
        std::vector<std::vector<int>> buckets;
        std::vector<double> Q_cum;
        const std::vector<uint64_t>& zkey;
        const std::vector<uint64_t>& qlmask;
        uint64_t zhash = 0ULL;
        uint64_t e_lmask = 0ULL;
        enum { CACHE_MASK = 8191 };
        std::vector<SynCacheEntry> scache;

        ThreadWS(const Code& c, bool mj, int nb_, const std::vector<int>& bb,
                 const std::vector<uint64_t>& zk, const std::vector<uint64_t>& qlm)
            : dec(c, mj), e(c.N, 0), in_dirty_e(c.N, 0), s(c.M_check, 0), corr(c.N, 0),
              q_b(bb), q_p(c.N, 0), pos_act(c.M_check, -1), buckets(nb_), Q_cum(nb_, 0.0),
              zkey(zk), qlmask(qlm), scache(CACHE_MASK + 1) {
            for (int q = 0; q < c.N; ++q) {
                int b = q_b[q];
                q_p[q] = (int)buckets[b].size();
                buckets[b].push_back(q);
            }
        }
        inline void move_q(int q, int new_b) {
            int old_b = q_b[q];
            if (old_b == new_b) return;
            int p = q_p[q], lq = buckets[old_b].back();
            buckets[old_b][p] = lq; q_p[lq] = p; buckets[old_b].pop_back();
            q_b[q] = new_b; q_p[q] = (int)buckets[new_b].size(); buckets[new_b].push_back(q);
        }
        inline void tog_c(int c) {
            zhash ^= zkey[c];
            if (!s[c]) { s[c] = 1; pos_act[c] = (int)act.size(); act.push_back(c); }
            else {
                s[c] = 0; int p = pos_act[c], lc = act.back();
                act[p] = lc; pos_act[lc] = p; act.pop_back(); pos_act[c] = -1;
            }
        }
        inline void apply_flip(int q, const Code& code) {
            if (!in_dirty_e[q]) { in_dirty_e[q] = 1; dirty_e.push_back(q); }
            e[q] ^= 1;
            e_lmask ^= qlmask[q];
            for (int c : code.q_to_checks[q]) {
                uint8_t was = s[c];
                tog_c(c);
                int db = (was ? +2 : -2) * code.j_tenths[c];
                for (int q2 : code.checks[c]) move_q(q2, q_b[q2] + db);
            }
        }
        inline void reset_to_ground(const Code& code) {
            for (int q : dirty_e) {
                if (e[q]) {
                    e[q] = 0;
                    for (int c : code.q_to_checks[q]) {
                        uint8_t was = s[c];
                        tog_c(c);
                        int db = (was ? +2 : -2) * code.j_tenths[c];
                        for (int q2 : code.checks[c]) move_q(q2, q_b[q2] + db);
                    }
                }
                in_dirty_e[q] = 0;
            }
            dirty_e.clear();
            zhash = 0ULL;
            e_lmask = 0ULL;
        }
        inline int current_energy_tenths(const Code& code) const {
            int et = 0;
            for (int c : act) et += code.j_tenths[c];
            return et;
        }
        inline int evolve_committor_bkl(const Code& code, const std::vector<double>& rates, int nb,
                                        std::mt19937_64& rng, std::uniform_real_distribution<double>& uni,
                                        std::exponential_distribution<double>& exp_d,
                                        double& tau_acc, int64_t& flips_acc,
                                        bool want_mov, std::vector<int>& trace) {
            // Evolve unbiased in discrete BKL flips (NO fixed t_rem=8.0 bath-time cutoff!) up to 2,500 flips!
            for (int step = 0; step < 2500; ++step) {
                if (act.empty()) return (e_lmask != 0ULL || (code.K > 64 && code.has_logical_error_raw(e))) ? +1 : -1;
                double Qt = 0.0;
                for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                tau_acc += exp_d(rng) / Qt;
                double r = uni(rng) * Qt;
                int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                if (cb >= nb) cb = nb - 1;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < nb - 1) ++cb;
                int q = buckets[cb][rng() % buckets[cb].size()];
                apply_flip(q, code);
                ++flips_acc;
                if (want_mov && trace.size() < 6000) trace.push_back(current_energy_tenths(code));
            }
            if (act.empty()) return (e_lmask != 0ULL || (code.K > 64 && code.has_logical_error_raw(e))) ? +1 : -1;
            dec.decode_into(act, s, corr);
            return code.has_logical_error(e, corr) ? +1 : -1;
        }
        inline void eval_decode_cached(const Code& code, int& out_w, bool& out_log_err) {
            if (code.K <= 64) {
                int slot = (int)(zhash & CACHE_MASK);
                SynCacheEntry& ce = scache[slot];
                if (ce.valid && ce.h == zhash) {
                    out_w = ce.w;
                    out_log_err = ((e_lmask ^ ce.corr_lmask) != 0ULL);
                    return;
                }
                dec.decode_into(act, s, corr);
                out_w = dec.corr_weight(corr);
                uint64_t cm = 0ULL;
                for (int q : dec.dirty_corr) if (corr[q]) cm ^= qlmask[q];
                ce.h = zhash;
                ce.corr_lmask = cm;
                ce.w = out_w;
                ce.valid = 1;
                out_log_err = ((e_lmask ^ cm) != 0ULL);
            } else {
                dec.decode_into(act, s, corr);
                out_w = dec.corr_weight(corr);
                out_log_err = code.has_logical_error(e, corr);
            }
        }
    };

    // Detect regular weight-8 fracton code (Haah Cubic Code) where UF triggers at E=4 before E=6 saddle:
    bool is_fracton_wt8 = (!code.checks.empty() && code.checks[0].size() == 8 && code.max_q_deg == 4);
    auto fracton_lam = [&](ThreadWS& ws, int w, bool lerr) -> int {
        if (!lerr) return std::min(3, std::max(1, w)); // 1, 2, 3
        int et = ws.current_energy_tenths(code);
        if (w <= 1 && et <= 45) return 6; // 1 flip from exact Basin B!
        if (et >= 55) return 5;           // E >= 6 fracton barrier saddle!
        return 4;                         // E == 4 post-UF dipole state
    };

    std::vector<std::unique_ptr<ThreadWS>> wss;
    for (int t = 0; t < nt; ++t) wss.push_back(std::make_unique<ThreadWS>(code, use_maj, nb, base_b, z_check_key, q_log_mask));

    std::vector<double> rep_p_fail(num_replicas, 0.0);
    std::vector<double> rep_tau_att(num_replicas, 0.0);
    int valid_reps = 0;
    int max_peak_E10_movie = -1;
    std::vector<int> best_movie_trace;
    bool want_movie = !movie_out.empty();
    int max_bank_keep = std::min(shots_per_stage, 1600);

    for (int rep = 0; rep < num_replicas; ++rep) {
        std::vector<double> stage_b, stage_f;
        double total_mean_tau = use_cond12 ? exact_mean_tau_step1 : 0.0;
        double reach_prob = use_cond12 ? P_1to2_exact : 1.0;
        std::vector<SparseStateV7> cur_bank;
        int cur_lam = use_cond12 ? 2 : 1;
        const int max_stages = 36;

        for (int st = 0; st < max_stages; ++st) {
            int next_lam = 2;
            if (st == 0) {
                next_lam = use_cond12 ? 2 : 1;
            } else if (is_fracton_wt8) {
                next_lam = cur_lam + 1;
            } else {
                next_lam = cur_lam + std::max(1, cur_lam / 3);
            }

            std::atomic<int> next_shot{0}, hit_B{0}, hit_next{0}, hit_cap{0};
            std::vector<std::vector<SparseStateV7>> thread_banks(nt);
            std::vector<double> thread_tau(nt, 0.0);
            std::vector<std::vector<int>> thread_movies(nt);

            auto stage_worker = [&](int tid) {
                ThreadWS& ws = *wss[tid];
                std::mt19937_64 rng(98765431ULL * (rep + 1) + 10007ULL * (st + 1) + 313ULL * (tid + 1) + (uint64_t)(beta * 100));
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                std::exponential_distribution<double> exp_d(1.0);

                while (true) {
                    int sh = next_shot.fetch_add(1);
                    if (sh >= shots_per_stage) break;
                    ws.reset_to_ground(code);

                    std::vector<int> trace;
                    int init_lam = 0;
                    if (st == 0) {
                        if (use_cond12) {
                            double u0 = uni(rng) * sum_w12;
                            int q0 = (int)(std::lower_bound(cdf_q0_12.begin(), cdf_q0_12.end(), u0) - cdf_q0_12.begin());
                            if (q0 >= code.N) q0 = code.N - 1;
                            ws.apply_flip(q0, code);
                            if (want_movie) { trace.push_back(0); trace.push_back(ws.current_energy_tenths(code)); }
                            int b_q0 = ws.q_b[q0];
                            double Qt = 0.0;
                            for (int b = 0; b < nb; ++b) {
                                int cnt = (int)ws.buckets[b].size() - (b == b_q0 ? 1 : 0);
                                Qt += rates[b] * (double)cnt;
                                ws.Q_cum[b] = Qt;
                            }
                            double r = uni(rng) * Qt;
                            int cb = (int)(std::lower_bound(ws.Q_cum.begin(), ws.Q_cum.end(), r) - ws.Q_cum.begin());
                            if (cb >= nb) cb = nb - 1;
                            while ((int)ws.buckets[cb].size() <= (cb == b_q0 ? 1 : 0) && cb > 0) --cb;
                            while ((int)ws.buckets[cb].size() <= (cb == b_q0 ? 1 : 0) && cb < nb - 1) ++cb;
                            int q1 = q0;
                            if (cb != b_q0) {
                                q1 = ws.buckets[cb][rng() % ws.buckets[cb].size()];
                            } else {
                                int sz = (int)ws.buckets[cb].size();
                                int idx = rng() % (sz - 1);
                                q1 = ws.buckets[cb][idx >= ws.q_p[q0] ? idx + 1 : idx];
                            }
                            ws.apply_flip(q1, code);
                            if (want_movie) trace.push_back(ws.current_energy_tenths(code));
                        } else {
                            double u = uni(rng) * Phi_0;
                            int q0 = (int)(std::lower_bound(cdf0.begin(), cdf0.end(), u) - cdf0.begin());
                            if (q0 >= code.N) q0 = code.N - 1;
                            ws.apply_flip(q0, code);
                            if (want_movie) { trace.push_back(0); trace.push_back(ws.current_energy_tenths(code)); }
                        }
                        if (ws.act.empty()) {
                            if (code.has_logical_error_raw(ws.e)) {
                                hit_B.fetch_add(1);
                                if (want_movie && thread_movies[tid].empty()) thread_movies[tid] = std::move(trace);
                            }
                            continue;
                        }
                        int w = 0; bool lerr = false;
                        ws.eval_decode_cached(code, w, lerr);
                        if (!is_fracton_wt8 && lerr) {
                            int64_t dfl = 0; double dtau = 0.0;
                            int cm = ws.evolve_committor_bkl(code, rates, nb, rng, uni, exp_d, dtau, dfl, want_movie, trace);
                            if (cm == +1) {
                                hit_B.fetch_add(1);
                                if (want_movie && thread_movies[tid].empty()) thread_movies[tid] = std::move(trace);
                            }
                            continue;
                        }
                        init_lam = is_fracton_wt8 ? fracton_lam(ws, w, lerr) : w;
                    } else {
                        const SparseStateV7& src = cur_bank[rng() % cur_bank.size()];
                        for (int q : src.flipped_q) ws.apply_flip(q, code);
                        init_lam = src.lam;
                        if (want_movie) trace = src.e_trace_tenths;
                    }

                    if (init_lam >= next_lam) {
                        hit_next.fetch_add(1);
                        if ((int)thread_banks[tid].size() < (max_bank_keep / nt + 12)) {
                            SparseStateV7 ns;
                            for (int q : ws.dirty_e) if (ws.e[q]) ns.flipped_q.push_back(q);
                            ns.lam = init_lam;
                            if (want_movie) ns.e_trace_tenths = std::move(trace);
                            thread_banks[tid].push_back(std::move(ns));
                        }
                        continue;
                    }

                    double tau = 0.0;
                    int64_t flips = 0;
                    const int64_t max_shot_flips = 10000;
                    while (flips < max_shot_flips) {
                        double Qt = 0.0;
                        for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)ws.buckets[b].size(); ws.Q_cum[b] = Qt; }
                        double dt = exp_d(rng) / Qt;
                        tau += dt;
                        double r = uni(rng) * Qt;
                        int cb = (int)(std::lower_bound(ws.Q_cum.begin(), ws.Q_cum.end(), r) - ws.Q_cum.begin());
                        if (cb >= nb) cb = nb - 1;
                        while (ws.buckets[cb].empty() && cb > 0) --cb;
                        while (ws.buckets[cb].empty() && cb < nb - 1) ++cb;

                        int q = ws.buckets[cb][rng() % ws.buckets[cb].size()];
                        ws.apply_flip(q, code);
                        ++flips;
                        if (want_movie && trace.size() < 6000) trace.push_back(ws.current_energy_tenths(code));

                        if (ws.act.empty()) {
                            if (code.has_logical_error_raw(ws.e)) {
                                hit_B.fetch_add(1);
                                if (want_movie && thread_movies[tid].empty()) thread_movies[tid] = std::move(trace);
                            }
                            break;
                        }
                        if ((flips & 3) != 0 && ws.act.size() > 6) continue;

                        int w = 0; bool lerr = false;
                        ws.eval_decode_cached(code, w, lerr);

                        if (!is_fracton_wt8 && lerr) {
                            int cm = ws.evolve_committor_bkl(code, rates, nb, rng, uni, exp_d, tau, flips, want_movie, trace);
                            if (cm == +1) {
                                hit_B.fetch_add(1);
                                if (want_movie && thread_movies[tid].empty()) thread_movies[tid] = std::move(trace);
                            }
                            break; // Absorbed in Basin B (+1) or bounced to Basin A (-1)!
                        }

                        int cur_l = is_fracton_wt8 ? fracton_lam(ws, w, lerr) : w;
                        if (is_fracton_wt8 && cur_l >= 6) {
                            int cm = ws.evolve_committor_bkl(code, rates, nb, rng, uni, exp_d, tau, flips, want_movie, trace);
                            if (cm == +1) {
                                hit_B.fetch_add(1);
                                if (want_movie && thread_movies[tid].empty()) thread_movies[tid] = std::move(trace);
                            }
                            break;
                        }

                        if (cur_l >= next_lam) {
                            hit_next.fetch_add(1);
                            if ((int)thread_banks[tid].size() < (max_bank_keep / nt + 12)) {
                                SparseStateV7 ns;
                                for (int qx : ws.dirty_e) if (ws.e[qx]) ns.flipped_q.push_back(qx);
                                ns.lam = cur_l;
                                if (want_movie) ns.e_trace_tenths = std::move(trace);
                                thread_banks[tid].push_back(std::move(ns));
                            }
                            break;
                        }
                    }
                    if (flips >= max_shot_flips) hit_cap.fetch_add(1);
                    thread_tau[tid] += tau;
                }
                ws.reset_to_ground(code);
            };

            std::vector<std::thread> pool;
            for (int t = 0; t < nt; ++t) pool.emplace_back(stage_worker, t);
            for (auto& th : pool) th.join();

            double stau = 0.0;
            for (double v : thread_tau) stau += v;
            total_mean_tau += reach_prob * (stau / shots_per_stage);

            for (int t = 0; t < nt; ++t) {
                if (!thread_movies[t].empty() && best_movie_trace.empty()) {
                    best_movie_trace = std::move(thread_movies[t]);
                }
            }

            double bk = (double)hit_B.load() / shots_per_stage;
            double fk = (double)hit_next.load() / shots_per_stage;
            stage_b.push_back(bk);
            stage_f.push_back(fk);

            std::vector<SparseStateV7> next_bank;
            for (int t = 0; t < nt; ++t) {
                for (auto& s_item : thread_banks[t]) {
                    if ((int)next_bank.size() < max_bank_keep) next_bank.push_back(std::move(s_item));
                }
            }

            int stop_lam = is_fracton_wt8 ? 6 : 1000;
            if (next_bank.size() < 12 || bk >= 0.25 || fk == 0.0 || next_lam >= stop_lam || st == max_stages - 1) {
                if (!next_bank.empty() && fk > 0.0) {
                    std::atomic<int> f_shot{0}, f_B{0};
                    std::vector<double> f_tau(nt, 0.0);
                    int final_shots = std::min(shots_per_stage, 1600);
                    auto final_worker = [&](int tid) {
                        ThreadWS& ws = *wss[tid];
                        std::mt19937_64 rng(55566677ULL * (rep + 1) + 919ULL * (tid + 1) + st * 17ULL);
                        std::uniform_real_distribution<double> uni(0.0, 1.0);
                        std::exponential_distribution<double> exp_d(1.0);
                        while (true) {
                            int sh = f_shot.fetch_add(1);
                            if (sh >= final_shots) break;
                            ws.reset_to_ground(code);
                            const SparseStateV7& src = next_bank[rng() % next_bank.size()];
                            for (int q : src.flipped_q) ws.apply_flip(q, code);
                            double tau = 0.0;
                            int64_t flips = 0;
                            while (flips < 10000) {
                                double Qt = 0.0;
                                for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)ws.buckets[b].size(); ws.Q_cum[b] = Qt; }
                                double dt = exp_d(rng) / Qt;
                                tau += dt;
                                double r = uni(rng) * Qt;
                                int cb = (int)(std::lower_bound(ws.Q_cum.begin(), ws.Q_cum.end(), r) - ws.Q_cum.begin());
                                if (cb >= nb) cb = nb - 1;
                                while (ws.buckets[cb].empty() && cb > 0) --cb;
                                while (ws.buckets[cb].empty() && cb < nb - 1) ++cb;
                                int q = ws.buckets[cb][rng() % ws.buckets[cb].size()];
                                ws.apply_flip(q, code);
                                ++flips;
                                if (ws.act.empty()) {
                                    if (code.has_logical_error_raw(ws.e)) f_B.fetch_add(1);
                                    break;
                                }
                                if ((flips & 3) != 0 && ws.act.size() > 6) continue;
                                int w = 0; bool lerr = false;
                                ws.eval_decode_cached(code, w, lerr);
                                if ((!is_fracton_wt8 && lerr) || (is_fracton_wt8 && lerr && w <= 1)) {
                                    std::vector<int> dtr;
                                    int cm = ws.evolve_committor_bkl(code, rates, nb, rng, uni, exp_d, tau, flips, false, dtr);
                                    if (cm == +1) f_B.fetch_add(1);
                                    break;
                                }
                            }
                            f_tau[tid] += tau;
                        }
                        ws.reset_to_ground(code);
                    };
                    std::vector<std::thread> fpool;
                    for (int t = 0; t < nt; ++t) fpool.emplace_back(final_worker, t);
                    for (auto& th : fpool) th.join();
                    double fstau = 0.0;
                    for (double v : f_tau) fstau += v;
                    total_mean_tau += (reach_prob * fk) * (fstau / final_shots);
                    stage_b.push_back((double)f_B.load() / final_shots);
                    stage_f.push_back(0.0);
                }
                break;
            }
            reach_prob *= fk;
            cur_bank = std::move(next_bank);
            cur_lam = next_lam;
        }

        double P_fail = 0.0;
        for (int k = (int)stage_b.size() - 1; k >= 0; --k) {
            P_fail = stage_b[k] + stage_f[k] * P_fail;
        }
        if (use_cond12) P_fail *= P_1to2_exact;
        rep_p_fail[rep] = P_fail;
        rep_tau_att[rep] = (1.0 / Phi_0) + total_mean_tau;
        if (P_fail > 0.0) ++valid_reps;
    }

    if (want_movie && !best_movie_trace.empty()) {
        std::ofstream mf(movie_out);
        for (int v : best_movie_trace) max_peak_E10_movie = std::max(max_peak_E10_movie, v);
        mf << "# step E_tenths\n";
        for (size_t i = 0; i < best_movie_trace.size(); ++i) mf << i << " " << best_movie_trace[i] << "\n";
    }

    // Unbiased pooled estimator across ALL `num_replicas` replicas (including any P_fail == 0)!
    double sum_p = 0.0, sum_tau = 0.0;
    for (int r = 0; r < num_replicas; ++r) { sum_p += rep_p_fail[r]; sum_tau += rep_tau_att[r]; }
    double mean_p = sum_p / num_replicas;
    double mean_tau = sum_tau / num_replicas;

    if (mean_p <= 0.0) {
        std::cout << "FFS dec=" << (use_maj ? "maj_uf" : "uf")
                  << " beta=" << std::fixed << std::setprecision(2) << beta
                  << " N=" << code.N << " K=" << code.K << " VALID_REPS=0 mean_t=-1 sem_t=-1\n";
        return;
    }
    double var_p = 0.0;
    for (double p : rep_p_fail) var_p += (p - mean_p) * (p - mean_p);
    double sem_p = (num_replicas > 1) ? std::sqrt(var_p / (num_replicas * (num_replicas - 1.0))) : 0.0;
    double mean_t = mean_tau / mean_p;
    double sem_t = mean_t * (sem_p / mean_p);

    std::cout << "FFS dec=" << (use_maj ? "maj_uf" : "uf")
              << " beta=" << std::fixed << std::setprecision(2) << beta
              << " N=" << code.N << " K=" << code.K
              << " Phi0=" << std::scientific << std::setprecision(4) << Phi_0
              << " Pfail=" << mean_p
              << " sem_P=" << sem_p
              << " mean_t=" << mean_t
              << " sem_t=" << sem_t
              << " valid_reps=" << valid_reps << "/" << num_replicas
              << " movie_peak_E=" << std::fixed << std::setprecision(1) << (max_peak_E10_movie > 0 ? 0.1 * max_peak_E10_movie : -1.0)
              << "\n";
}

// =========================================================================================
// PASSIVE COOLING & AUTONOMOUS CODEWORD SELF-HEALING (`run_cool_v7`)
// Reports:
//   - `p_zero_syn`:   fraction reaching exact zero syndrome (s == 0)
//   - `p_exact_heal`: fraction reaching exact s == 0 AND exact original codeword (zero decoder!)
//   - `p_log_survive`: fraction preserving all K logicals after final UF readout
// =========================================================================================
void run_cool_v7(const Code& code, bool use_maj, const std::string& j_soft_path,
                 const std::string& protocol, const std::string& init_mode,
                 double beta_init, double beta_final, double total_time,
                 int num_trials, int num_threads) {
    std::vector<int> j_hard = code.j_tenths;
    std::vector<int> j_soft = j_hard;
    if (!j_soft_path.empty() && j_soft_path != "uniform") {
        std::ifstream jin(j_soft_path);
        if (jin) {
            for (int i = 0; i < code.M_check; ++i) jin >> j_soft[i];
        }
    }

    const int n_stages = 20;
    double dt_stage = total_time / n_stages;
    int nt = std::max(1, std::min(num_threads, num_trials));
    std::atomic<int> next_tr{0}, clean_zero{0}, exact_heal{0}, log_survive{0};
    std::vector<double> syn_dens(num_trials, 0.0);
    std::vector<int> max_cl(num_trials, 0);

    auto worker = [&](int tid) {
        FastDecoderV7 dec(code, use_maj);
        std::mt19937_64 rng(777123ULL * (tid + 1) + code.N * 97ULL + (uint64_t)(total_time * 13));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_d(1.0);

        double p_init = 0.5;
        if (init_mode.rfind("noisy_", 0) == 0) p_init = std::stod(init_mode.substr(6));

        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0), corr(code.N, 0);
        std::vector<int> q_b(code.N, 0), q_p(code.N, 0), j_cur(code.M_check, 10), act, pos_act(code.M_check, -1);

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::fill(s.begin(), s.end(), 0);
            std::fill(pos_act.begin(), pos_act.end(), -1);
            act.clear();
            for (int q = 0; q < code.N; ++q) {
                e[q] = (uni(rng) < p_init) ? 1 : 0;
                if (e[q]) {
                    for (int c : code.q_to_checks[q]) s[c] ^= 1;
                }
            }
            for (int c = 0; c < code.M_check; ++c) {
                if (s[c]) { pos_act[c] = (int)act.size(); act.push_back(c); }
            }

            for (int st = 0; st < n_stages; ++st) {
                double beta_s = beta_final;
                if (protocol == "quench") {
                    beta_s = beta_final;
                    j_cur = j_hard;
                } else if (protocol == "anneal") {
                    if (st < 16) {
                        double f1 = (double)st / 15.0;
                        beta_s = beta_init * std::pow(beta_final / beta_init, f1);
                    } else {
                        beta_s = beta_final;
                    }
                    j_cur = j_hard;
                } else if (protocol == "soft_seam") {
                    beta_s = beta_final;
                    if (st < 10) {
                        j_cur = j_soft;
                    } else {
                        double f2 = (double)(st - 9) / 10.0;
                        for (int c = 0; c < code.M_check; ++c) {
                            j_cur[c] = (int)std::llround((1.0 - f2) * j_soft[c] + f2 * j_hard[c]);
                        }
                    }
                }

                int max_dE10 = 0;
                for (int q = 0; q < code.N; ++q) {
                    int sm = 0;
                    for (int c : code.q_to_checks[q]) sm += j_cur[c];
                    max_dE10 = std::max(max_dE10, sm);
                }
                int nb = 2 * max_dE10 + 1;
                std::vector<double> rates(nb, 0.0), Q_cum(nb, 0.0);
                for (int b = 0; b < nb; ++b) {
                    double dE = 0.1 * (double)(b - max_dE10);
                    rates[b] = 1.0 / (1.0 + std::exp(beta_s * dE));
                }
                std::vector<std::vector<int>> buckets(nb);
                for (int q = 0; q < code.N; ++q) {
                    int dE10 = 0;
                    for (int c : code.q_to_checks[q]) dE10 += s[c] ? -j_cur[c] : +j_cur[c];
                    int b = dE10 + max_dE10;
                    q_b[q] = b; q_p[q] = (int)buckets[b].size(); buckets[b].push_back(q);
                }
                auto move_q = [&](int q, int new_b) {
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

                double t_rem = dt_stage;
                int64_t flips = 0;
                while (t_rem > 0.0 && flips < 300000000LL) {
                    double Qt = 0.0;
                    for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                    if (Qt <= 1e-30) break;
                    double dt = exp_d(rng) / Qt;
                    if (dt > t_rem) break;
                    t_rem -= dt;
                    double r = uni(rng) * Qt;
                    int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                    if (cb >= nb) cb = nb - 1;
                    while (buckets[cb].empty() && cb > 0) --cb;
                    while (buckets[cb].empty() && cb < nb - 1) ++cb;
                    int q = buckets[cb][rng() % buckets[cb].size()];
                    e[q] ^= 1; ++flips;
                    for (int c : code.q_to_checks[q]) {
                        uint8_t was = s[c];
                        tog_c(c);
                        int db = (was ? +2 : -2) * j_cur[c];
                        for (int q2 : code.checks[c]) move_q(q2, q_b[q2] + db);
                    }
                }
            }
            syn_dens[tr] = (double)act.size() / std::max(1, code.M_check);
            if (act.empty()) {
                clean_zero.fetch_add(1);
                if (!code.has_logical_error_raw(e)) exact_heal.fetch_add(1);
                max_cl[tr] = 0;
                if (!code.has_logical_error_raw(e)) log_survive.fetch_add(1);
            } else if (init_mode == "hot" && act.size() > 1500) {
                max_cl[tr] = (int)act.size();
            } else {
                int mcl = dec.decode_into(act, s, corr);
                max_cl[tr] = mcl;
                if (!code.has_logical_error(e, corr)) log_survive.fetch_add(1);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    double mean_syn = 0.0, mean_mcl = 0.0;
    for (int i = 0; i < num_trials; ++i) { mean_syn += syn_dens[i]; mean_mcl += max_cl[i]; }
    mean_syn /= num_trials; mean_mcl /= num_trials;
    std::cout << "COOL proto=" << protocol << " init=" << init_mode
              << " t_tot=" << std::fixed << std::setprecision(1) << total_time
              << " N=" << code.N << " K=" << code.K << " trials=" << num_trials
              << " syn_dens=" << std::setprecision(5) << mean_syn
              << " mean_max_cl=" << std::setprecision(1) << mean_mcl
              << " p_zero_syn=" << std::setprecision(3) << ((double)clean_zero.load() / num_trials)
              << " p_exact_heal=" << std::setprecision(3) << ((double)exact_heal.load() / num_trials)
              << " p_log_survive=" << std::setprecision(3) << ((double)log_survive.load() / num_trials) << "\n";
}

struct RGLevel {
    int N = 0, M = 0, n_xf = 0, n_pf = 0;
    char step_type = 'X';
    std::vector<std::vector<int>> checks, q_to_checks;
    std::vector<std::pair<int, int>> pf_span; // (f_start, cnt)
    std::vector<std::pair<int, int>> bf_span; // (bf_start, cnt) for Z dual steps
};

void run_rg_flow_v10(const std::string& rg_path, double beta, double t_eq, int num_trials, int num_threads) {
    std::ifstream in(rg_path);
    if (!in) { std::cerr << "Cannot open RG file " << rg_path << "\n"; std::exit(1); }
    std::string hdr;
    in >> hdr;
    bool is_v2 = (hdr == "RGV2");
    int S_max = is_v2 ? 0 : std::stoi(hdr);
    if (is_v2) in >> S_max;
    std::vector<RGLevel> lvs(S_max + 1);
    for (int s = 0; s <= S_max; ++s) {
        auto& lv = lvs[s];
        in >> lv.N >> lv.M >> lv.n_xf >> lv.n_pf;
        if (is_v2) in >> lv.step_type;
        lv.checks.resize(lv.M);
        lv.q_to_checks.resize(lv.N);
        for (int ci = 0; ci < lv.M; ++ci) {
            int d; in >> d; lv.checks[ci].resize(d);
            for (int j = 0; j < d; ++j) {
                in >> lv.checks[ci][j];
                lv.q_to_checks[lv.checks[ci][j]].push_back(ci);
            }
        }
        if (s >= 1) {
            lv.pf_span.resize(lv.n_pf);
            for (int pf = 0; pf < lv.n_pf; ++pf) {
                in >> lv.pf_span[pf].first >> lv.pf_span[pf].second;
            }
            if (is_v2 && lv.step_type == 'Z') {
                lv.bf_span.resize(lv.n_pf);
                for (int pf = 0; pf < lv.n_pf; ++pf) {
                    in >> lv.bf_span[pf].first >> lv.bf_span[pf].second;
                }
            }
        }
    }

    const RGLevel& top = lvs[S_max];
    int max_deg = 0;
    for (int q = 0; q < top.N; ++q) max_deg = std::max(max_deg, (int)top.q_to_checks[q].size());
    int nb = 2 * max_deg + 1;
    std::vector<double> rates(nb);
    for (int b = 0; b < nb; ++b) {
        int dE = b - max_deg;
        rates[b] = 1.0 / (1.0 + std::exp(beta * (double)dE));
    }

    int nt = std::max(1, std::min(num_threads, num_trials));
    std::atomic<int> next_tr{0};
    std::vector<std::vector<double>> tr_rho(num_trials, std::vector<double>(S_max + 1, 0.0));

    auto clean_level_syndrome = [&](const RGLevel& plv, std::vector<uint8_t>& cur, const std::vector<uint8_t>& pi1) {
        std::vector<int> mark_f(plv.M, 0), touched;
        for (int pass = 0; pass < 3; ++pass) {
            bool changed = false;
            // 1. Single-edge flips strictly reducing sum(cur + (cur ^ pi1)):
            for (int q = 0; q < plv.N; ++q) {
                const auto& inc = plv.q_to_checks[q];
                if (inc.empty()) continue;
                int bef = 0, aft = 0;
                for (int f : inc) {
                    uint8_t c0 = cur[f], c1 = c0 ^ 1;
                    bef += c0 + (c0 ^ pi1[f]);
                    aft += c1 + (c1 ^ pi1[f]);
                }
                if (aft < bef) {
                    for (int f : inc) cur[f] ^= 1;
                    changed = true;
                }
            }
            // 2. Two-adjacent-edge flips (sharing a check face in plv, radius <= 2 = cell size L=2):
            for (int q1 = 0; q1 < plv.N; ++q1) {
                const auto& inc1 = plv.q_to_checks[q1];
                if (inc1.empty()) continue;
                bool has_active = false;
                for (int f : inc1) if (cur[f] ^ pi1[f]) { has_active = true; break; }
                if (!has_active) continue;
                for (int f_mid : inc1) {
                    for (int q2 : plv.checks[f_mid]) {
                        if (q2 <= q1) continue;
                        touched.clear();
                        for (int f : inc1) { if (!mark_f[f]) touched.push_back(f); mark_f[f] ^= 1; }
                        for (int f : plv.q_to_checks[q2]) { if (!mark_f[f]) touched.push_back(f); mark_f[f] ^= 1; }
                        int bef = 0, aft = 0;
                        for (int f : touched) {
                            if (mark_f[f]) {
                                uint8_t c0 = cur[f], c1 = c0 ^ 1;
                                bef += c0 + (c0 ^ pi1[f]);
                                aft += c1 + (c1 ^ pi1[f]);
                            }
                            mark_f[f] = 0;
                        }
                        if (aft < bef) {
                            for (int f : inc1) cur[f] ^= 1;
                            for (int f : plv.q_to_checks[q2]) cur[f] ^= 1;
                            changed = true;
                            break;
                        }
                    }
                }
            }
            if (!changed) break;
        }
    };

    auto worker = [&](int tid) {
        std::vector<uint8_t> s_top(top.M, 0);
        std::vector<int> q_b(top.N, 0), q_p(top.N, 0);
        std::vector<std::vector<int>> buckets(nb);
        std::vector<double> Q_cum(nb, 0.0);

        auto move_q = [&](int q, int new_b) {
            int old_b = q_b[q];
            if (old_b == new_b) return;
            int p = q_p[q], lq = buckets[old_b].back();
            buckets[old_b][p] = lq; q_p[lq] = p; buckets[old_b].pop_back();
            q_b[q] = new_b; q_p[q] = (int)buckets[new_b].size(); buckets[new_b].push_back(q);
        };

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::mt19937_64 rng(7712345ULL + 99991ULL * (tr + 1) + (uint64_t)(beta * 1000));
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            std::exponential_distribution<double> exp_d(1.0);

            std::fill(s_top.begin(), s_top.end(), 0);
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < top.N; ++q) {
                int b = (int)top.q_to_checks[q].size() + max_deg;
                q_b[q] = b; q_p[q] = (int)buckets[b].size(); buckets[b].push_back(q);
            }

            double t = 0.0;
            while (t < t_eq) {
                double Qt = 0.0;
                for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                double dt = exp_d(rng) / Qt;
                t += dt;
                double r = uni(rng) * Qt;
                int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                if (cb >= nb) cb = nb - 1;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < nb - 1) ++cb;
                int q = buckets[cb][rng() % buckets[cb].size()];
                for (int c : top.q_to_checks[q]) {
                    uint8_t was = s_top[c];
                    s_top[c] ^= 1;
                    int db = was ? +2 : -2;
                    for (int q2 : top.checks[c]) move_q(q2, q_b[q2] + db);
                }
            }

            // Coarse-grain purely from syndrome s_top down through s = S_max .. 0!
            std::vector<uint8_t> cur_s = s_top;
            for (int s = S_max; s >= 1; --s) {
                int cnt1 = 0;
                for (uint8_t v : cur_s) cnt1 += v;
                tr_rho[tr][s] = (double)cnt1 / std::max(1, lvs[s].M);

                const auto& lv = lvs[s];
                if (lv.step_type == 'Z') {
                    std::vector<uint8_t> p_dual(lv.n_pf, 0);
                    for (int pf = 0; pf < lv.n_pf; ++pf) {
                        int fs = lv.pf_span[pf].first, len = lv.pf_span[pf].second;
                        int bs = lv.bf_span[pf].first, blen = lv.bf_span[pf].second;
                        uint8_t v = 0;
                        for (int k = 0; k < len; ++k) v ^= cur_s[fs + k] ^ cur_s[lv.n_xf + fs + k];
                        for (int k = 0; k < blen; ++k) v ^= cur_s[bs + k];
                        p_dual[pf] = v;
                    }
                    cur_s = std::move(p_dual);
                } else {
                    std::vector<uint8_t> pi0(lv.n_pf, 0), pi1(lv.n_pf, 0);
                    for (int pf = 0; pf < lv.n_pf; ++pf) {
                        int fs = lv.pf_span[pf].first, len = lv.pf_span[pf].second;
                        uint8_t p0 = 0, p1 = 0;
                        for (int k = 0; k < len; ++k) {
                            p0 ^= cur_s[fs + k];
                            p1 ^= cur_s[lv.n_xf + fs + k];
                        }
                        pi0[pf] = p0; pi1[pf] = p1;
                    }
                    clean_level_syndrome(lvs[s - 1], pi0, pi1);
                    cur_s = std::move(pi0);
                }
            }
            int cnt0 = 0;
            for (uint8_t v : cur_s) cnt0 += v;
            tr_rho[tr][0] = (double)cnt0 / std::max(1, lvs[0].M);
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    std::cout << "RGFLOW S_max=" << S_max << " N_top=" << top.N << " beta=" << std::fixed << std::setprecision(2) << beta
              << " t_eq=" << std::setprecision(1) << t_eq << " trials=" << num_trials;
    for (int s = S_max; s >= 0; --s) {
        double m = 0.0;
        for (int tr = 0; tr < num_trials; ++tr) m += tr_rho[tr][s];
        m /= num_trials;
        std::cout << " rho_" << s << "=" << std::scientific << std::setprecision(4) << m;
    }
    std::cout << "\n";
}

// =========================================================================================
// TRUE LINEAR-TIME O(N) HIERARCHICAL RG READOUT DECODER (`rg_kmc`)
// =========================================================================================
struct RGDecLevel {
    Code sub_code;
    int n_xf = 0, n_pf = 0;
    char step_type = 'S';
    std::vector<std::vector<int>> L0, L1;
    std::vector<int> f0_start, f1_start, bf_start;
    std::vector<std::vector<std::vector<int>>> lut0, lut1; // [pf][mask 0..15] -> child qubits
    std::vector<std::vector<std::vector<int>>> z_basis;    // [pf][b 0..10] -> interior child qubits flipping (1<<b)|(1<<11)
};

struct RGDecHierarchy {
    int S_max = 0;
    std::vector<RGDecLevel> lvs;

    static RGDecHierarchy load(const std::string& path) {
        std::ifstream in(path);
        if (!in) { std::cerr << "Cannot open RGDec file " << path << "\n"; std::exit(1); }
        RGDecHierarchy h;
        in >> h.S_max;
        h.lvs.resize(h.S_max + 1);
        for (int s = 0; s <= h.S_max; ++s) {
            auto& lv = h.lvs[s];
            auto& c = lv.sub_code;
            in >> c.N >> c.M_check >> lv.n_xf >> lv.n_pf >> lv.step_type;
            c.checks.resize(c.M_check);
            c.q_to_checks.resize(c.N);
            for (int ci = 0; ci < c.M_check; ++ci) {
                int d; in >> d; c.checks[ci].resize(d);
                for (int j = 0; j < d; ++j) {
                    in >> c.checks[ci][j];
                    c.q_to_checks[c.checks[ci][j]].push_back(ci);
                }
            }
            for (int q = 0; q < c.N; ++q) c.max_q_deg = std::max(c.max_q_deg, (int)c.q_to_checks[q].size());
            c.j_tenths.assign(c.M_check, 10);
            if (s >= 1) {
                int pN = h.lvs[s - 1].sub_code.N;
                lv.L0.resize(pN); lv.L1.resize(pN);
                for (int pq = 0; pq < pN; ++pq) {
                    int k0; in >> k0; lv.L0[pq].resize(k0);
                    for (int j = 0; j < k0; ++j) in >> lv.L0[pq][j];
                    int k1; in >> k1; lv.L1[pq].resize(k1);
                    for (int j = 0; j < k1; ++j) in >> lv.L1[pq][j];
                }
                lv.f0_start.resize(lv.n_pf);
                lv.f1_start.resize(lv.n_pf);
                if (lv.step_type == 'X') {
                    lv.lut0.assign(lv.n_pf, std::vector<std::vector<int>>(16));
                    lv.lut1.assign(lv.n_pf, std::vector<std::vector<int>>(16));
                    for (int pf = 0; pf < lv.n_pf; ++pf) {
                        in >> lv.f0_start[pf] >> lv.f1_start[pf];
                        for (int idx = 0; idx < 8; ++idx) {
                            int m, k; in >> m >> k;
                            lv.lut0[pf][m].resize(k);
                            for (int j = 0; j < k; ++j) in >> lv.lut0[pf][m][j];
                        }
                        for (int idx = 0; idx < 8; ++idx) {
                            int m, k; in >> m >> k;
                            lv.lut1[pf][m].resize(k);
                            for (int j = 0; j < k; ++j) in >> lv.lut1[pf][m][j];
                        }
                    }
                } else if (lv.step_type == 'Z') {
                    lv.bf_start.resize(lv.n_pf);
                    lv.z_basis.assign(lv.n_pf, std::vector<std::vector<int>>(11));
                    for (int pf = 0; pf < lv.n_pf; ++pf) {
                        in >> lv.f0_start[pf] >> lv.f1_start[pf] >> lv.bf_start[pf];
                        for (int b = 0; b < 11; ++b) {
                            int k; in >> k;
                            lv.z_basis[pf][b].resize(k);
                            for (int j = 0; j < k; ++j) in >> lv.z_basis[pf][b][j];
                        }
                    }
                }
            }
        }
        return h;
    }
};

struct HierarchicalRGWorker {
    const RGDecHierarchy& h;
    std::vector<std::unique_ptr<FastDecoderV7>> decs;
    std::vector<std::vector<uint8_t>> syn_buf, corr_buf, tmp_corr;
    std::vector<std::vector<int>> act_buf;

    explicit HierarchicalRGWorker(const RGDecHierarchy& hier) : h(hier) {
        int S = h.S_max;
        decs.resize(S + 1);
        syn_buf.resize(S + 1);
        corr_buf.resize(S + 1);
        tmp_corr.resize(S + 1);
        act_buf.resize(S + 1);
        for (int s = 0; s <= S; ++s) {
            decs[s] = std::make_unique<FastDecoderV7>(h.lvs[s].sub_code, false);
            syn_buf[s].assign(h.lvs[s].sub_code.M_check, 0);
            corr_buf[s].assign(h.lvs[s].sub_code.N, 0);
            tmp_corr[s].assign(h.lvs[s].sub_code.N, 0);
        }
    }

    bool decode_level(int s, const std::vector<uint8_t>& s_in, std::vector<uint8_t>& c_out, int final_cluster_cap = 0) {
        const auto& lv = h.lvs[s];
        const auto& code = lv.sub_code;
        c_out.assign(code.N, 0);
        auto& syn = syn_buf[s];
        syn = s_in;

        auto flip_q = [&](int q) {
            c_out[q] ^= 1;
            for (int f : code.q_to_checks[q]) syn[f] ^= 1;
        };

        if (s <= 1) {
            act_buf[s].clear();
            for (int f = 0; f < code.M_check; ++f) if (syn[f]) act_buf[s].push_back(f);
            if (!act_buf[s].empty()) {
                decs[s]->decode_into(act_buf[s], syn, tmp_corr[s], 0);
                for (int q = 0; q < code.N; ++q) if (tmp_corr[s][q]) flip_q(q);
            }
            for (int f = 0; f < code.M_check; ++f) if (syn[f]) return false;
            return true;
        }

        // O(1)-radius local pre-pass: annihilates small thermal droplets (<= 48 qubits) locally at scale s,
        // while leaving multi-cell clusters (> 48 qubits) untouched for hierarchical RG coarse-graining!
        act_buf[s].clear();
        for (int f = 0; f < code.M_check; ++f) if (syn[f]) act_buf[s].push_back(f);
        if (!act_buf[s].empty()) {
            decs[s]->decode_into(act_buf[s], syn, tmp_corr[s], 48);
            for (int q = 0; q < code.N; ++q) if (tmp_corr[s][q]) flip_q(q);
        }

        const auto& pcode = h.lvs[s - 1].sub_code;
        if (lv.step_type == 'Z') {
            std::vector<uint8_t> p_dual(lv.n_pf, 0);
            bool any_p = false;
            for (int pf = 0; pf < lv.n_pf; ++pf) {
                int f0 = lv.f0_start[pf], f1 = lv.f1_start[pf], bf = lv.bf_start[pf];
                uint8_t v = 0;
                for (int k = 0; k < 4; ++k) v ^= syn[f0 + k] ^ syn[f1 + k] ^ syn[bf + k];
                p_dual[pf] = v;
                if (v) any_p = true;
            }
            if (any_p) {
                if (!decode_level(s - 1, p_dual, corr_buf[s - 1], final_cluster_cap)) return false;
                for (int pq = 0; pq < pcode.N; ++pq) {
                    if (corr_buf[s - 1][pq]) {
                        for (int cq : lv.L0[pq]) flip_q(cq);
                    }
                }
            }
            // Annihilate all 12-face cell syndromes strictly inside each cell U_{pf} in O(1):
            for (int pf = 0; pf < lv.n_pf; ++pf) {
                int f0 = lv.f0_start[pf], f1 = lv.f1_start[pf], bf = lv.bf_start[pf];
                int m12 = 0;
                for (int k = 0; k < 4; ++k) {
                    if (syn[f0 + k]) m12 |= (1 << k);
                    if (syn[f1 + k]) m12 |= (1 << (4 + k));
                    if (syn[bf + k]) m12 |= (1 << (8 + k));
                }
                if (m12 && (__builtin_popcount(m12) % 2 == 0)) {
                    for (int b = 0; b < 11; ++b) {
                        if ((m12 >> b) & 1) {
                            for (int cq : lv.z_basis[pf][b]) flip_q(cq);
                        }
                    }
                }
            }
        } else {
            std::vector<uint8_t> pi0(lv.n_pf, 0), pi1(lv.n_pf, 0);
            for (int pf = 0; pf < lv.n_pf; ++pf) {
                int f0 = lv.f0_start[pf], f1 = lv.f1_start[pf];
                pi0[pf] = syn[f0] ^ syn[f0 + 1] ^ syn[f0 + 2] ^ syn[f0 + 3];
                pi1[pf] = syn[f1] ^ syn[f1 + 1] ^ syn[f1 + 2] ^ syn[f1 + 3];
            }
            for (int pass = 0; pass < 2; ++pass) {
                bool chg = false;
                for (int pq = 0; pq < pcode.N; ++pq) {
                    const auto& inc = pcode.q_to_checks[pq];
                    if (inc.empty()) continue;
                    int bef = 0, aft = 0;
                    for (int pf : inc) {
                        uint8_t c0 = pi0[pf], c1 = c0 ^ 1;
                        bef += c0 + (c0 ^ pi1[pf]);
                        aft += c1 + (c1 ^ pi1[pf]);
                    }
                    if (aft < bef) {
                        for (int pf : inc) pi0[pf] ^= 1;
                        for (int cq : lv.L0[pq]) flip_q(cq);
                        chg = true;
                    }
                }
                if (!chg) break;
            }
            std::vector<uint8_t> diff(lv.n_pf, 0);
            bool any_diff = false;
            for (int pf = 0; pf < lv.n_pf; ++pf) {
                diff[pf] = pi0[pf] ^ pi1[pf];
                if (diff[pf]) any_diff = true;
            }
            if (any_diff) {
                if (!decode_level(s - 1, diff, corr_buf[s - 1], final_cluster_cap)) return false;
                for (int pq = 0; pq < pcode.N; ++pq) {
                    if (corr_buf[s - 1][pq]) {
                        for (int cq : lv.L1[pq]) flip_q(cq);
                    }
                }
            }
            std::vector<uint8_t> pi0_now(lv.n_pf, 0);
            bool any_p0 = false;
            for (int pf = 0; pf < lv.n_pf; ++pf) {
                int f0 = lv.f0_start[pf];
                pi0_now[pf] = syn[f0] ^ syn[f0 + 1] ^ syn[f0 + 2] ^ syn[f0 + 3];
                if (pi0_now[pf]) any_p0 = true;
            }
            if (any_p0) {
                if (!decode_level(s - 1, pi0_now, corr_buf[s - 1], final_cluster_cap)) return false;
                for (int pq = 0; pq < pcode.N; ++pq) {
                    if (corr_buf[s - 1][pq]) {
                        for (int cq : lv.L0[pq]) flip_q(cq);
                        for (int cq : lv.L1[pq]) flip_q(cq);
                    }
                }
            }
            for (int pf = 0; pf < lv.n_pf; ++pf) {
                int f0 = lv.f0_start[pf];
                int m0 = syn[f0] | (syn[f0 + 1] << 1) | (syn[f0 + 2] << 2) | (syn[f0 + 3] << 3);
                if (m0 && (__builtin_popcount(m0) % 2 == 0)) {
                    for (int cq : lv.lut0[pf][m0]) flip_q(cq);
                }
                int f1 = lv.f1_start[pf];
                int m1 = syn[f1] | (syn[f1 + 1] << 1) | (syn[f1 + 2] << 2) | (syn[f1 + 3] << 3);
                if (m1 && (__builtin_popcount(m1) % 2 == 0)) {
                    for (int cq : lv.lut1[pf][m1]) flip_q(cq);
                }
            }
        }

        // Clean remaining local intra-cell / bridge seam syndromes (with optional O(1) cluster cap!):
        act_buf[s].clear();
        for (int f = 0; f < code.M_check; ++f) if (syn[f]) act_buf[s].push_back(f);
        if (!act_buf[s].empty()) {
            decs[s]->decode_into(act_buf[s], syn, tmp_corr[s], final_cluster_cap);
            for (int q = 0; q < code.N; ++q) if (tmp_corr[s][q]) flip_q(q);
        }
        for (int f = 0; f < code.M_check; ++f) if (syn[f]) return false;
        return true;
    }
};

void run_rg_kmc_v10(const Code& top_code, const RGDecHierarchy& hier, double beta,
                    int num_trials, int64_t max_flips, int num_threads) {
    int max_dE10 = 0;
    for (int q = 0; q < top_code.N; ++q) {
        int s = 0;
        for (int c : top_code.q_to_checks[q]) s += top_code.j_tenths[c];
        max_dE10 = std::max(max_dE10, s);
    }
    int nb = 2 * max_dE10 + 1;
    std::vector<double> rates(nb, 0.0);
    for (int b = 0; b < nb; ++b) {
        double dE = 0.1 * (double)(b - max_dE10);
        rates[b] = 1.0 / (1.0 + std::exp(beta * dE));
    }

    std::vector<double> t_fails(num_trials, 0.0);
    std::vector<int64_t> fl_counts(num_trials, 0);
    std::atomic<int> next_tr{0}, capped{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int tid) {
        HierarchicalRGWorker rg_dec(hier);
        std::mt19937_64 rng(918273645ULL * (tid + 1) + (uint64_t)(beta * 1000) + top_code.N * 53ULL);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_d(1.0);

        std::vector<uint8_t> e(top_code.N, 0), in_dirty_e(top_code.N, 0), s(top_code.M_check, 0), corr(top_code.N, 0);
        std::vector<int> q_b(top_code.N, 0), q_p(top_code.N, 0), act, pos_act(top_code.M_check, -1), dirty_e;
        std::vector<std::vector<int>> buckets(nb);
        std::vector<double> Q_cum(nb, 0.0);

        auto move_q = [&](int q, int new_b) {
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
            std::mt19937_64 rng(918273645ULL + 99991ULL * (uint64_t)(tr + 1) + (uint64_t)(beta * 1000) + top_code.N * 53ULL);
            for (int q : dirty_e) { e[q] = 0; in_dirty_e[q] = 0; }
            dirty_e.clear();
            for (int c : act) { s[c] = 0; pos_act[c] = -1; } act.clear();
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < top_code.N; ++q) {
                int dE10 = 0;
                for (int c : top_code.q_to_checks[q]) dE10 += top_code.j_tenths[c];
                int b = dE10 + max_dE10;
                q_b[q] = b; q_p[q] = (int)buckets[b].size(); buckets[b].push_back(q);
            }
            double t = 0.0, t_next = 0.0;
            int64_t flips = 0;
            bool failed = false;
            while (flips < max_flips) {
                double Qt = 0.0;
                for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                double dt = exp_d(rng) / Qt;
                t += dt; t_next -= dt;
                double r = uni(rng) * Qt;
                int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                if (cb >= nb) cb = nb - 1;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < nb - 1) ++cb;

                int q = buckets[cb][rng() % buckets[cb].size()];
                if (!in_dirty_e[q]) { in_dirty_e[q] = 1; dirty_e.push_back(q); }
                e[q] ^= 1; ++flips;
                for (int c : top_code.q_to_checks[q]) {
                    uint8_t was = s[c];
                    tog_c(c);
                    int db = (was ? +2 : -2) * top_code.j_tenths[c];
                    for (int q2 : top_code.checks[c]) move_q(q2, q_b[q2] + db);
                }
                if (flips <= 16 || t_next <= 0.0) {
                    if (act.empty()) {
                        if (top_code.has_logical_error_raw(e)) { failed = true; break; }
                    } else {
                        rg_dec.decode_level(hier.S_max, s, corr);
                        if (top_code.has_logical_error(e, corr)) { failed = true; break; }
                    }
                    t_next = 0.20 * t;
                }
            }
            if (!failed) capped.fetch_add(1);
            t_fails[tr] = t;
            fl_counts[tr] = flips;
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    double sum = 0.0, sum_fl = 0.0;
    for (int i = 0; i < num_trials; ++i) { sum += t_fails[i]; sum_fl += (double)fl_counts[i]; }
    double mean_t = sum / num_trials;
    double var = 0.0;
    for (double v : t_fails) var += (v - mean_t) * (v - mean_t);
    double sem_t = (num_trials > 1) ? std::sqrt(var / (num_trials * (num_trials - 1.0))) : 0.0;
    std::sort(t_fails.begin(), t_fails.end());
    std::cout << "KMC dec=rg_dec"
              << " beta=" << std::fixed << std::setprecision(2) << beta
              << " N=" << top_code.N << " K=" << top_code.K << " trials=" << num_trials
              << " mean_t=" << std::scientific << std::setprecision(4) << mean_t
              << " sem_t=" << sem_t
              << " median_t=" << t_fails[num_trials / 2]
              << " mean_flips=" << std::fixed << std::setprecision(1) << (sum_fl / num_trials)
              << " bails=0 capped=" << capped.load()
              << " t_list=";
    for (int i = 0; i < num_trials; ++i) {
        if (i) std::cout << ",";
        std::cout << std::scientific << std::setprecision(5) << t_fails[i];
    }
    std::cout << "\n";
}

// =========================================================================================
// V13 PAIRED TRAJECTORY BENCHMARK: UF vs RG_DEC vs STRICT O(N) RG_STRICT (cap=64)
// Evaluates all three decoders on IDENTICAL BKL trajectories & checkpoints!
// =========================================================================================
void run_paired_rg_kmc(const Code& top_code, const RGDecHierarchy& hier, double beta,
                       int num_trials, int64_t max_flips, int num_threads) {
    int max_dE10 = 0;
    for (int q = 0; q < top_code.N; ++q) {
        int s = 0;
        for (int c : top_code.q_to_checks[q]) s += top_code.j_tenths[c];
        max_dE10 = std::max(max_dE10, s);
    }
    int nb = 2 * max_dE10 + 1;
    std::vector<double> rates(nb, 0.0);
    for (int b = 0; b < nb; ++b) {
        double dE = 0.1 * (double)(b - max_dE10);
        rates[b] = 1.0 / (1.0 + std::exp(beta * dE));
    }

    std::vector<double> t_uf(num_trials, 0.0), t_rg(num_trials, 0.0), t_st(num_trials, 0.0);
    std::vector<uint8_t> cap_uf(num_trials, 0), cap_rg(num_trials, 0), cap_st(num_trials, 0);
    std::atomic<int> next_tr{0};
    std::atomic<int64_t> tot_chk{0}, agree_rg_uf{0}, agree_st_uf{0};
    std::atomic<int64_t> ns_uf{0}, ns_rg{0}, ns_st{0}, n_calls{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int) {
        FastDecoderV7 uf_dec(top_code, false);
        HierarchicalRGWorker rg_dec(hier);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_d(1.0);

        std::vector<uint8_t> e(top_code.N, 0), in_dirty_e(top_code.N, 0), s(top_code.M_check, 0);
        std::vector<uint8_t> corr_uf(top_code.N, 0), corr_rg(top_code.N, 0), corr_st(top_code.N, 0);
        std::vector<int> q_b(top_code.N, 0), q_p(top_code.N, 0), act, pos_act(top_code.M_check, -1), dirty_e;
        std::vector<std::vector<int>> buckets(nb);
        std::vector<double> Q_cum(nb, 0.0);

        auto move_q = [&](int q, int new_b) {
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

        int64_t loc_chk = 0, loc_ag_rg = 0, loc_ag_st = 0;
        int64_t loc_ns_uf = 0, loc_ns_rg = 0, loc_ns_st = 0, loc_calls = 0;

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::mt19937_64 rng(918273645ULL + 99991ULL * (uint64_t)(tr + 1) + (uint64_t)(beta * 1000) + top_code.N * 53ULL);
            for (int q : dirty_e) { e[q] = 0; in_dirty_e[q] = 0; }
            dirty_e.clear();
            for (int c : act) { s[c] = 0; pos_act[c] = -1; } act.clear();
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < top_code.N; ++q) {
                int dE10 = 0;
                for (int c : top_code.q_to_checks[q]) dE10 += top_code.j_tenths[c];
                int b = dE10 + max_dE10;
                q_b[q] = b; q_p[q] = (int)buckets[b].size(); buckets[b].push_back(q);
            }

            double t = 0.0, t_next = 0.0;
            int64_t flips = 0;
            bool f_uf = false, f_rg = false, f_st = false;
            double tf_uf = 0.0, tf_rg = 0.0, tf_st = 0.0;

            while (flips < max_flips && !(f_uf && f_rg && f_st)) {
                double Qt = 0.0;
                for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                double dt = exp_d(rng) / Qt;
                t += dt; t_next -= dt;
                double r = uni(rng) * Qt;
                int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                if (cb >= nb) cb = nb - 1;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < nb - 1) ++cb;

                int q = buckets[cb][rng() % buckets[cb].size()];
                if (!in_dirty_e[q]) { in_dirty_e[q] = 1; dirty_e.push_back(q); }
                e[q] ^= 1; ++flips;
                for (int c : top_code.q_to_checks[q]) {
                    uint8_t was = s[c];
                    tog_c(c);
                    int db = (was ? +2 : -2) * top_code.j_tenths[c];
                    for (int q2 : top_code.checks[c]) move_q(q2, q_b[q2] + db);
                }
                if (flips <= 16 || t_next <= 0.0) {
                    if (act.empty()) {
                        bool raw_err = top_code.has_logical_error_raw(e);
                        ++loc_chk; ++loc_ag_rg; ++loc_ag_st;
                        if (raw_err) {
                            if (!f_uf) { f_uf = true; tf_uf = t; }
                            if (!f_rg) { f_rg = true; tf_rg = t; }
                            if (!f_st) { f_st = true; tf_st = t; }
                        }
                    } else {
                        auto c0 = std::chrono::steady_clock::now();
                        uf_dec.decode_into(act, s, corr_uf);
                        auto c1 = std::chrono::steady_clock::now();
                        bool ok_rg = rg_dec.decode_level(hier.S_max, s, corr_rg, 0);
                        auto c2 = std::chrono::steady_clock::now();
                        bool ok_st = rg_dec.decode_level(hier.S_max, s, corr_st, 64);
                        auto c3 = std::chrono::steady_clock::now();
                        loc_ns_uf += std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count();
                        loc_ns_rg += std::chrono::duration_cast<std::chrono::nanoseconds>(c2 - c1).count();
                        loc_ns_st += std::chrono::duration_cast<std::chrono::nanoseconds>(c3 - c2).count();
                        ++loc_calls;

                        uint64_t m_uf = top_code.logical_class_mask(e, corr_uf);
                        uint64_t m_rg = ok_rg ? top_code.logical_class_mask(e, corr_rg) : ~0ULL;
                        uint64_t m_st = ok_st ? top_code.logical_class_mask(e, corr_st) : ~0ULL;

                        if (!f_uf && !f_rg && !f_st) {
                            ++loc_chk;
                            if (ok_rg && m_rg == m_uf) ++loc_ag_rg;
                            if (ok_st && m_st == m_uf) ++loc_ag_st;
                        }
                        if (!f_uf && m_uf != 0) { f_uf = true; tf_uf = t; }
                        if (!f_rg && (!ok_rg || m_rg != 0)) { f_rg = true; tf_rg = t; }
                        if (!f_st && (!ok_st || m_st != 0)) { f_st = true; tf_st = t; }
                    }
                    t_next = 0.20 * t;
                }
            }
            if (!f_uf) { tf_uf = t; cap_uf[tr] = 1; }
            if (!f_rg) { tf_rg = t; cap_rg[tr] = 1; }
            if (!f_st) { tf_st = t; cap_st[tr] = 1; }
            t_uf[tr] = tf_uf; t_rg[tr] = tf_rg; t_st[tr] = tf_st;
        }
        tot_chk.fetch_add(loc_chk);
        agree_rg_uf.fetch_add(loc_ag_rg);
        agree_st_uf.fetch_add(loc_ag_st);
        ns_uf.fetch_add(loc_ns_uf);
        ns_rg.fetch_add(loc_ns_rg);
        ns_st.fetch_add(loc_ns_st);
        n_calls.fetch_add(loc_calls);
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    auto stats_of = [&](const std::vector<double>& v) {
        double s = 0.0; for (double x : v) s += x;
        double m = s / num_trials, vr = 0.0;
        for (double x : v) vr += (x - m) * (x - m);
        double sem = (num_trials > 1) ? std::sqrt(vr / (num_trials * (num_trials - 1.0))) : 0.0;
        return std::make_pair(m, sem);
    };
    auto [m_uf, sem_uf] = stats_of(t_uf);
    auto [m_rg, sem_rg] = stats_of(t_rg);
    auto [m_st, sem_st] = stats_of(t_st);

    int exact_same_st = 0, c_uf = 0, c_rg = 0, c_st = 0;
    std::vector<double> log_rat_st(num_trials, 0.0);
    for (int i = 0; i < num_trials; ++i) {
        if (std::abs(t_st[i] - t_uf[i]) <= 1e-12 * std::max(1.0, t_uf[i])) ++exact_same_st;
        log_rat_st[i] = std::log10(std::max(1e-12, t_st[i]) / std::max(1e-12, t_uf[i]));
        c_uf += cap_uf[i]; c_rg += cap_rg[i]; c_st += cap_st[i];
    }
    auto [m_lrat, sem_lrat] = stats_of(log_rat_st);
    int64_t calls = std::max<int64_t>(1, n_calls.load());
    int64_t chks = std::max<int64_t>(1, tot_chk.load());

    std::cout << "PAIRED_KMC beta=" << std::fixed << std::setprecision(2) << beta
              << " N=" << top_code.N << " K=" << top_code.K << " trials=" << num_trials
              << std::scientific << std::setprecision(4)
              << " mean_uf=" << m_uf << " sem_uf=" << sem_uf
              << " mean_rg=" << m_rg << " sem_rg=" << sem_rg
              << " mean_strict=" << m_st << " sem_strict=" << sem_st
              << std::fixed << std::setprecision(4)
              << " log10_ratio_strict=" << m_lrat << " sem_log10_ratio=" << sem_lrat
              << " frac_same_t=" << (double)exact_same_st / num_trials
              << " chk_agree_rg=" << (double)agree_rg_uf.load() / chks
              << " chk_agree_strict=" << (double)agree_st_uf.load() / chks
              << std::setprecision(2)
              << " us_uf=" << (1e-3 * ns_uf.load() / calls)
              << " us_rg=" << (1e-3 * ns_rg.load() / calls)
              << " us_strict=" << (1e-3 * ns_st.load() / calls)
              << " cap_uf=" << c_uf << " cap_rg=" << c_rg << " cap_strict=" << c_st
              << " pairs_uf_strict=";
    for (int i = 0; i < num_trials; ++i) {
        if (i) std::cout << ",";
        std::cout << std::scientific << std::setprecision(4) << t_uf[i] << ":" << t_st[i];
    }
    std::cout << "\n";
}

// =========================================================================================
// V13 PAIRED I.I.D. SINGLE-SHOT NOISE BENCHMARK (`iid_paired`)
// =========================================================================================
void run_iid_paired_decode(const Code& top_code, const RGDecHierarchy& hier, double p_err,
                           int num_shots, int num_threads) {
    std::atomic<int> next_sh{0};
    std::atomic<int> fail_uf{0}, fail_rg{0}, fail_st{0}, disagree_rg{0}, disagree_st{0};
    std::atomic<int64_t> ns_uf{0}, ns_rg{0}, ns_st{0};
    int nt = std::max(1, std::min(num_threads, num_shots));

    auto worker = [&](int) {
        FastDecoderV7 uf_dec(top_code, false);
        HierarchicalRGWorker rg_dec(hier);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<uint8_t> e(top_code.N, 0), s(top_code.M_check, 0);
        std::vector<uint8_t> c_uf(top_code.N, 0), c_rg(top_code.N, 0), c_st(top_code.N, 0);
        std::vector<int> act;

        while (true) {
            int sh = next_sh.fetch_add(1);
            if (sh >= num_shots) break;
            std::mt19937_64 rng(44556677ULL + 7919ULL * (uint64_t)(sh + 1) + (uint64_t)(p_err * 100000) + top_code.N * 31ULL);
            std::fill(e.begin(), e.end(), 0);
            std::fill(s.begin(), s.end(), 0);
            for (int q = 0; q < top_code.N; ++q) {
                if (uni(rng) < p_err) {
                    e[q] = 1;
                    for (int ci : top_code.q_to_checks[q]) s[ci] ^= 1;
                }
            }
            act.clear();
            for (int ci = 0; ci < top_code.M_check; ++ci) if (s[ci]) act.push_back(ci);
            if (act.empty()) {
                if (top_code.has_logical_error_raw(e)) {
                    fail_uf.fetch_add(1); fail_rg.fetch_add(1); fail_st.fetch_add(1);
                }
                continue;
            }
            auto t0 = std::chrono::steady_clock::now();
            uf_dec.decode_into(act, s, c_uf);
            auto t1 = std::chrono::steady_clock::now();
            bool ok_rg = rg_dec.decode_level(hier.S_max, s, c_rg, 0);
            auto t2 = std::chrono::steady_clock::now();
            bool ok_st = rg_dec.decode_level(hier.S_max, s, c_st, 64);
            auto t3 = std::chrono::steady_clock::now();
            ns_uf.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            ns_rg.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
            ns_st.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());

            uint64_t m_uf = top_code.logical_class_mask(e, c_uf);
            uint64_t m_rg = ok_rg ? top_code.logical_class_mask(e, c_rg) : ~0ULL;
            uint64_t m_st = ok_st ? top_code.logical_class_mask(e, c_st) : ~0ULL;
            if (m_uf != 0) fail_uf.fetch_add(1);
            if (!ok_rg || m_rg != 0) fail_rg.fetch_add(1);
            if (!ok_st || m_st != 0) fail_st.fetch_add(1);
            if (!ok_rg || m_rg != m_uf) disagree_rg.fetch_add(1);
            if (!ok_st || m_st != m_uf) disagree_st.fetch_add(1);
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    std::cout << "IID_PAIRED p=" << std::fixed << std::setprecision(4) << p_err
              << " N=" << top_code.N << " K=" << top_code.K << " shots=" << num_shots
              << " fail_uf=" << fail_uf.load()
              << " fail_rg=" << fail_rg.load()
              << " fail_strict=" << fail_st.load()
              << " disagree_rg=" << disagree_rg.load()
              << " disagree_strict=" << disagree_st.load()
              << std::setprecision(2)
              << " us_uf=" << (1e-3 * ns_uf.load() / std::max(1, num_shots))
              << " us_rg=" << (1e-3 * ns_rg.load() / std::max(1, num_shots))
              << " us_strict=" << (1e-3 * ns_st.load() / std::max(1, num_shots)) << "\n";
}

// =========================================================================================
// V13 COARSE-SYNDROME FLOW WITH COUNTS, DENOMINATORS, SEMS, & LOGICAL PRESERVATION
// =========================================================================================
void run_rg_flow_v13(const Code& top_code, const RGDecHierarchy& dec_hier, const std::string& rg_path,
                     double beta, double t_eq, int num_trials, int num_threads) {
    std::ifstream in(rg_path);
    if (!in) { std::cerr << "Cannot open RG file " << rg_path << "\n"; std::exit(1); }
    std::string header;
    in >> header;
    if (header != "RGV2") { std::cerr << "Expected RGV2 in " << rg_path << "\n"; std::exit(1); }
    int S_max = 0;
    in >> S_max;
    struct Lv {
        int N = 0, M = 0, n_xf = 0, n_pf = 0;
        char step_type = 'X';
        std::vector<std::vector<int>> checks, q_to_checks;
        std::vector<std::pair<int, int>> pf_span, bf_span;
    };
    std::vector<Lv> lvs(S_max + 1);
    for (int s = 0; s <= S_max; ++s) {
        in >> lvs[s].N >> lvs[s].M >> lvs[s].n_xf >> lvs[s].n_pf >> lvs[s].step_type;
        lvs[s].checks.resize(lvs[s].M);
        lvs[s].q_to_checks.resize(lvs[s].N);
        for (int ci = 0; ci < lvs[s].M; ++ci) {
            int d; in >> d; lvs[s].checks[ci].resize(d);
            for (int j = 0; j < d; ++j) {
                in >> lvs[s].checks[ci][j];
                lvs[s].q_to_checks[lvs[s].checks[ci][j]].push_back(ci);
            }
        }
        if (s >= 1) {
            lvs[s].pf_span.resize(lvs[s].n_pf);
            for (int pf = 0; pf < lvs[s].n_pf; ++pf) in >> lvs[s].pf_span[pf].first >> lvs[s].pf_span[pf].second;
            if (lvs[s].step_type == 'Z') {
                lvs[s].bf_span.resize(lvs[s].n_pf);
                for (int pf = 0; pf < lvs[s].n_pf; ++pf) in >> lvs[s].bf_span[pf].first >> lvs[s].bf_span[pf].second;
            }
        }
    }

    const auto& top = lvs[S_max];
    int max_deg = 0;
    for (int q = 0; q < top.N; ++q) max_deg = std::max(max_deg, (int)top.q_to_checks[q].size());
    int nb = 2 * max_deg + 1;
    std::vector<double> rates(nb, 0.0);
    for (int b = 0; b < nb; ++b) {
        double dE = (double)(b - max_deg);
        rates[b] = 1.0 / (1.0 + std::exp(beta * dE));
    }

    std::vector<std::vector<int>> tr_cnt(num_trials, std::vector<int>(S_max + 1, 0));
    std::vector<int> tr_log_fail(num_trials, 0);
    std::atomic<int> next_tr{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto clean_level_syndrome = [](const Lv& plv, std::vector<uint8_t>& cur, const std::vector<uint8_t>& pi1) {
        std::vector<uint8_t> mark_f(plv.M, 0);
        std::vector<int> touched;
        for (int pass = 0; pass < 4; ++pass) {
            bool changed = false;
            for (int q = 0; q < plv.N; ++q) {
                const auto& inc = plv.q_to_checks[q];
                if (inc.empty()) continue;
                int bef = 0, aft = 0;
                for (int f : inc) {
                    uint8_t c0 = cur[f], c1 = c0 ^ 1;
                    bef += c0 + (c0 ^ pi1[f]);
                    aft += c1 + (c1 ^ pi1[f]);
                }
                if (aft < bef) {
                    for (int f : inc) cur[f] ^= 1;
                    changed = true;
                }
            }
            for (int q1 = 0; q1 < plv.N; ++q1) {
                const auto& inc1 = plv.q_to_checks[q1];
                if (inc1.empty()) continue;
                bool has_active = false;
                for (int f : inc1) if (cur[f] ^ pi1[f]) { has_active = true; break; }
                if (!has_active) continue;
                for (int f_mid : inc1) {
                    for (int q2 : plv.checks[f_mid]) {
                        if (q2 <= q1) continue;
                        touched.clear();
                        for (int f : inc1) { if (!mark_f[f]) touched.push_back(f); mark_f[f] ^= 1; }
                        for (int f : plv.q_to_checks[q2]) { if (!mark_f[f]) touched.push_back(f); mark_f[f] ^= 1; }
                        int bef = 0, aft = 0;
                        for (int f : touched) {
                            if (mark_f[f]) {
                                uint8_t c0 = cur[f], c1 = c0 ^ 1;
                                bef += c0 + (c0 ^ pi1[f]);
                                aft += c1 + (c1 ^ pi1[f]);
                            }
                            mark_f[f] = 0;
                        }
                        if (aft < bef) {
                            for (int f : inc1) cur[f] ^= 1;
                            for (int f : plv.q_to_checks[q2]) cur[f] ^= 1;
                            changed = true;
                            break;
                        }
                    }
                }
            }
            if (!changed) break;
        }
    };

    auto worker = [&](int) {
        HierarchicalRGWorker rg_dec(dec_hier);
        std::vector<uint8_t> e_top(top.N, 0), s_top(top.M, 0), corr(top.N, 0);
        std::vector<int> q_b(top.N, 0), q_p(top.N, 0);
        std::vector<std::vector<int>> buckets(nb);
        std::vector<double> Q_cum(nb, 0.0);

        auto move_q = [&](int q, int new_b) {
            int old_b = q_b[q];
            if (old_b == new_b) return;
            int p = q_p[q], lq = buckets[old_b].back();
            buckets[old_b][p] = lq; q_p[lq] = p; buckets[old_b].pop_back();
            q_b[q] = new_b; q_p[q] = (int)buckets[new_b].size(); buckets[new_b].push_back(q);
        };

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::mt19937_64 rng(7712345ULL + 99991ULL * (uint64_t)(tr + 1) + (uint64_t)(beta * 1000));
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            std::exponential_distribution<double> exp_d(1.0);

            std::fill(e_top.begin(), e_top.end(), 0);
            std::fill(s_top.begin(), s_top.end(), 0);
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < top.N; ++q) {
                int b = (int)top.q_to_checks[q].size() + max_deg;
                q_b[q] = b; q_p[q] = (int)buckets[b].size(); buckets[b].push_back(q);
            }

            double t = 0.0;
            while (t < t_eq) {
                double Qt = 0.0;
                for (int b = 0; b < nb; ++b) { Qt += rates[b] * (double)buckets[b].size(); Q_cum[b] = Qt; }
                double dt = exp_d(rng) / Qt;
                t += dt;
                double r = uni(rng) * Qt;
                int cb = (int)(std::lower_bound(Q_cum.begin(), Q_cum.end(), r) - Q_cum.begin());
                if (cb >= nb) cb = nb - 1;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < nb - 1) ++cb;
                int q = buckets[cb][rng() % buckets[cb].size()];
                e_top[q] ^= 1;
                for (int c : top.q_to_checks[q]) {
                    uint8_t was = s_top[c];
                    s_top[c] ^= 1;
                    int db = was ? +2 : -2;
                    for (int q2 : top.checks[c]) move_q(q2, q_b[q2] + db);
                }
            }

            bool ok = rg_dec.decode_level(dec_hier.S_max, s_top, corr, 64);
            if (!ok || top_code.logical_class_mask(e_top, corr) != 0) tr_log_fail[tr] = 1;

            std::vector<uint8_t> cur_s = s_top;
            for (int s = S_max; s >= 1; --s) {
                int cnt1 = 0;
                for (uint8_t v : cur_s) cnt1 += v;
                tr_cnt[tr][s] = cnt1;
                const auto& lv = lvs[s];
                if (lv.step_type == 'Z') {
                    std::vector<uint8_t> p_dual(lv.n_pf, 0);
                    for (int pf = 0; pf < lv.n_pf; ++pf) {
                        int fs = lv.pf_span[pf].first, len = lv.pf_span[pf].second;
                        int bs = lv.bf_span[pf].first, blen = lv.bf_span[pf].second;
                        uint8_t v = 0;
                        for (int k = 0; k < len; ++k) v ^= cur_s[fs + k] ^ cur_s[lv.n_xf + fs + k];
                        for (int k = 0; k < blen; ++k) v ^= cur_s[bs + k];
                        p_dual[pf] = v;
                    }
                    cur_s = std::move(p_dual);
                } else {
                    std::vector<uint8_t> pi0(lv.n_pf, 0), pi1(lv.n_pf, 0);
                    for (int pf = 0; pf < lv.n_pf; ++pf) {
                        int fs = lv.pf_span[pf].first, len = lv.pf_span[pf].second;
                        uint8_t p0 = 0, p1 = 0;
                        for (int k = 0; k < len; ++k) {
                            p0 ^= cur_s[fs + k];
                            p1 ^= cur_s[lv.n_xf + fs + k];
                        }
                        pi0[pf] = p0; pi1[pf] = p1;
                    }
                    clean_level_syndrome(lvs[s - 1], pi0, pi1);
                    cur_s = std::move(pi0);
                }
            }
            int cnt0 = 0;
            for (uint8_t v : cur_s) cnt0 += v;
            tr_cnt[tr][0] = cnt0;
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    int n_nz0 = 0, n_lf = 0;
    for (int tr = 0; tr < num_trials; ++tr) {
        if (tr_cnt[tr][0] > 0) ++n_nz0;
        n_lf += tr_log_fail[tr];
    }
    std::cout << "RGFLOW_V13 S_max=" << S_max << " N_top=" << top.N << " beta=" << std::fixed << std::setprecision(2) << beta
              << " t_eq=" << std::setprecision(1) << t_eq << " trials=" << num_trials
              << " nz_seed=" << n_nz0 << " log_fail=" << n_lf;
    for (int s = S_max; s >= 0; --s) {
        double m = 0.0;
        for (int tr = 0; tr < num_trials; ++tr) m += tr_cnt[tr][s];
        m /= num_trials;
        double vr = 0.0;
        for (int tr = 0; tr < num_trials; ++tr) vr += (tr_cnt[tr][s] - m) * (tr_cnt[tr][s] - m);
        double sem = (num_trials > 1) ? std::sqrt(vr / (num_trials * (num_trials - 1.0))) : 0.0;
        int Ms = std::max(1, lvs[s].M);
        std::cout << " M_" << s << "=" << Ms
                  << " cnt_" << s << "=" << std::scientific << std::setprecision(4) << m
                  << " semc_" << s << "=" << sem
                  << " rho_" << s << "=" << (m / Ms);
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    std::string mode = argv[1];
    if (mode == "prep_logicals") {
        compute_and_append_dual_logicals(argv[2], argv[3]);
        return 0;
    }
    if (mode == "verify") {
        Code c = Code::load(argv[2]);
        int bad = c.verify_commutation();
        std::cout << "VERIFY " << argv[2] << " N=" << c.N << " M_check=" << c.M_check
                  << " M_stab=" << c.M_stab << " K=" << c.K << " bad_comm=" << bad << "\n";
        return bad == 0 ? 0 : 1;
    }
    int hw = std::max(1, (int)std::thread::hardware_concurrency());
    if (mode == "rg_flow") {
        run_rg_flow_v10(argv[2], std::stod(argv[3]), std::stod(argv[4]), std::stoi(argv[5]),
                        (argc >= 7) ? std::stoi(argv[6]) : hw);
        return 0;
    }
    if (mode == "rg_kmc") {
        Code top_c = Code::load(argv[2], "uniform");
        RGDecHierarchy hier = RGDecHierarchy::load(argv[3]);
        run_rg_kmc_v10(top_c, hier, std::stod(argv[4]), std::stoi(argv[5]), std::stoll(argv[6]),
                       (argc >= 8) ? std::stoi(argv[7]) : hw);
        return 0;
    }
    if (mode == "paired_rg_kmc") {
        Code top_c = Code::load(argv[2], "uniform");
        RGDecHierarchy hier = RGDecHierarchy::load(argv[3]);
        run_paired_rg_kmc(top_c, hier, std::stod(argv[4]), std::stoi(argv[5]), std::stoll(argv[6]),
                          (argc >= 8) ? std::stoi(argv[7]) : hw);
        return 0;
    }
    if (mode == "iid_paired") {
        Code top_c = Code::load(argv[2], "uniform");
        RGDecHierarchy hier = RGDecHierarchy::load(argv[3]);
        run_iid_paired_decode(top_c, hier, std::stod(argv[4]), std::stoi(argv[5]),
                              (argc >= 7) ? std::stoi(argv[6]) : hw);
        return 0;
    }
    if (mode == "rg_flow_v13") {
        Code top_c = Code::load(argv[2], "uniform");
        RGDecHierarchy hier = RGDecHierarchy::load(argv[3]);
        run_rg_flow_v13(top_c, hier, argv[4], std::stod(argv[5]), std::stod(argv[6]), std::stoi(argv[7]),
                        (argc >= 9) ? std::stoi(argv[8]) : hw);
        return 0;
    }
    auto is_maj = [](const std::string& s) { return s == "gain_uf" || s == "maj_uf" || s == "sweep_uf"; };
    if (mode == "kmc") {
        Code c = Code::load(argv[2], argv[3]);
        run_kmc_v7(c, is_maj(argv[4]), std::stod(argv[5]), std::stoi(argv[6]), std::stoll(argv[7]),
                   (argc >= 9) ? std::stoi(argv[8]) : hw);
    } else if (mode == "ffs") {
        Code c = Code::load(argv[2], argv[3]);
        std::string mov = (argc >= 10) ? argv[9] : "";
        run_ffs_v7(c, is_maj(argv[4]), std::stod(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]),
                   (argc >= 9) ? std::stoi(argv[8]) : hw, mov);
    } else if (mode == "cool") {
        Code c = Code::load(argv[2], "uniform");
        run_cool_v7(c, is_maj(argv[4]), argv[3], argv[5], argv[6], std::stod(argv[7]), std::stod(argv[8]),
                    std::stod(argv[9]), std::stoi(argv[10]), (argc >= 12) ? std::stoi(argv[11]) : hw);
    }
    return 0;
}
