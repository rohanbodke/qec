// kmc_engine.cpp
// High-performance C++20 analyzer & rejection-free Kinetic Monte Carlo (n-fold way Glauber)
// simulator for 3D Self-Correcting Quantum Memories (BDL arXiv:2605.10943, Layer Codes arXiv:2510.06659).
//
// Modes:
//   1. stats     : Exact/ISD distances and minimax energy barriers
//   2. threshold : i.i.d. Pauli error channel P_L(p) across decoders (uf, bposd, rg_strict, rg_relaxed)
//   3. kmc       : Continuous-time rejection-free Glauber thermal bath memory lifetime <t_fail(beta)>
//   4. cool      : Passive thermal initialization / cooling from random state (BDL Outlook Open Problem #2)
//   5. hybrid    : Hybrid lazy active QEC + continuous passive thermal bath (logical error rate vs tau_EC)

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct RGTree {
    // levels[0] = finest unit partition (scale 1), ..., levels.back() = coarsest before seed
    // levels[lv][u] = list of check indices in unit u at scale lv
    std::vector<std::vector<std::vector<int>>> levels;
    bool valid = false;

    static RGTree load_optional(const std::string& code_path) {
        RGTree rg;
        std::string rg_path = code_path + ".rg";
        std::ifstream in(rg_path);
        if (!in) return rg;
        int num_levels = 0;
        if (!(in >> num_levels) || num_levels <= 0) return rg;
        rg.levels.resize(num_levels);
        for (int lv = 0; lv < num_levels; ++lv) {
            int num_units = 0;
            in >> num_units;
            rg.levels[lv].resize(num_units);
            for (int u = 0; u < num_units; ++u) {
                int sz = 0;
                in >> sz;
                rg.levels[lv][u].resize(sz);
                for (int j = 0; j < sz; ++j) in >> rg.levels[lv][u][j];
            }
        }
        rg.valid = true;
        return rg;
    }
};

struct Code {
    int N = 0;
    int M_check = 0;
    int M_stab = 0;
    int K = 0;
    int max_q_deg = 0;
    std::vector<std::vector<int>> checks;
    std::vector<std::vector<int>> q_to_checks;
    std::vector<std::vector<int>> stabs;
    std::vector<std::vector<int>> logicals;
    RGTree rg;

    static Code load(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            std::cerr << "Failed to open code file: " << path << "\n";
            std::exit(1);
        }
        Code c;
        in >> c.N >> c.M_check >> c.M_stab >> c.K;
        c.checks.resize(c.M_check);
        c.q_to_checks.resize(c.N);
        for (int i = 0; i < c.M_check; ++i) {
            int d;
            in >> d;
            c.checks[i].resize(d);
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
            int d;
            in >> d;
            c.stabs[i].resize(d);
            for (int j = 0; j < d; ++j) in >> c.stabs[i][j];
        }
        c.logicals.resize(c.K);
        for (int i = 0; i < c.K; ++i) {
            int d;
            in >> d;
            c.logicals[i].resize(d);
            for (int j = 0; j < d; ++j) in >> c.logicals[i][j];
        }
        c.rg = RGTree::load_optional(path);
        return c;
    }

    bool has_logical_error(const std::vector<uint8_t>& e, const std::vector<uint8_t>& corr) const {
        for (int a = 0; a < K; ++a) {
            uint8_t parity = 0;
            for (int q : logicals[a]) {
                parity ^= (e[q] ^ corr[q]);
            }
            if (parity) return true;
        }
        return false;
    }

    bool verify_syndrome(const std::vector<uint8_t>& s, const std::vector<uint8_t>& corr) const {
        for (int c = 0; c < M_check; ++c) {
            uint8_t p = 0;
            for (int q : checks[c]) p ^= corr[q];
            if (p != s[c]) return false;
        }
        return true;
    }
};

struct BitRow {
    std::vector<uint64_t> w;
    BitRow() = default;
    explicit BitRow(int nbits) : w((nbits + 63) / 64, 0ULL) {}
    inline void flip(int b) { w[b >> 6] ^= (1ULL << (b & 63)); }
    inline bool get(int b) const { return (w[b >> 6] >> (b & 63)) & 1ULL; }
    inline void xor_with(const BitRow& o) {
        for (size_t i = 0; i < w.size(); ++i) w[i] ^= o.w[i];
    }
    inline int popcount() const {
        int s = 0;
        for (uint64_t x : w) s += __builtin_popcountll(x);
        return s;
    }
};

bool solve_gf2_submatrix(
    int nr, int nc,
    const std::vector<std::vector<int>>& row_cols,
    const std::vector<uint8_t>& rhs,
    std::vector<uint8_t>& sol_out,
    bool min_weight_refine = true
) {
    int width = nc + 1;
    std::vector<BitRow> mat(nr, BitRow(width));
    for (int r = 0; r < nr; ++r) {
        for (int c : row_cols[r]) mat[r].flip(c);
        if (rhs[r]) mat[r].flip(nc);
    }

    std::vector<int> pivot_col_of_row(nr, -1);
    std::vector<int> pivot_row_of_col(nc, -1);
    int rank = 0;
    for (int c = 0; c < nc && rank < nr; ++c) {
        int sel = -1;
        for (int r = rank; r < nr; ++r) {
            if (mat[r].get(c)) {
                sel = r;
                break;
            }
        }
        if (sel == -1) continue;
        if (sel != rank) std::swap(mat[sel], mat[rank]);
        pivot_col_of_row[rank] = c;
        pivot_row_of_col[c] = rank;
        for (int r = 0; r < nr; ++r) {
            if (r != rank && mat[r].get(c)) {
                mat[r].xor_with(mat[rank]);
            }
        }
        ++rank;
    }

    for (int r = rank; r < nr; ++r) {
        if (mat[r].get(nc)) return false;
    }

    sol_out.assign(nc, 0);
    for (int r = 0; r < rank; ++r) {
        if (mat[r].get(nc)) sol_out[pivot_col_of_row[r]] = 1;
    }

    if (min_weight_refine && nc > rank) {
        BitRow cur(nc);
        for (int c = 0; c < nc; ++c) if (sol_out[c]) cur.flip(c);
        int cur_w = cur.popcount();

        std::vector<BitRow> null_vecs;
        for (int c = 0; c < nc && (int)null_vecs.size() < 48; ++c) {
            if (pivot_row_of_col[c] != -1) continue;
            BitRow nv(nc);
            nv.flip(c);
            for (int r = 0; r < rank; ++r) {
                if (mat[r].get(c)) nv.flip(pivot_col_of_row[r]);
            }
            null_vecs.push_back(std::move(nv));
        }

        bool improved = true;
        int passes = 0;
        while (improved && passes < 4) {
            improved = false;
            ++passes;
            for (const auto& nv : null_vecs) {
                int w2 = 0;
                for (size_t i = 0; i < cur.w.size(); ++i) {
                    w2 += __builtin_popcountll(cur.w[i] ^ nv.w[i]);
                }
                if (w2 < cur_w) {
                    cur.xor_with(nv);
                    cur_w = w2;
                    improved = true;
                }
            }
        }
        for (int c = 0; c < nc; ++c) sol_out[c] = cur.get(c) ? 1 : 0;
    }
    return true;
}

// ============================================================================
// DECODER 1: ClusterUF (Gu-Kubica arXiv:2510.06659 Section 2.1 Cluster Decoder)
// ============================================================================
struct ClusterUFDecoder {
    const Code& code;
    explicit ClusterUFDecoder(const Code& c) : code(c) {}

    std::vector<uint8_t> decode(const std::vector<uint8_t>& s) const {
        std::vector<uint8_t> corr(code.N, 0);
        std::vector<int> active_syns;
        for (int c = 0; c < code.M_check; ++c) if (s[c]) active_syns.push_back(c);
        if (active_syns.empty()) return corr;

        int V = code.M_check + code.N;
        std::vector<int> parent(V, -1);
        std::vector<uint8_t> cluster_solved(V, 0);
        std::vector<std::vector<int>> cl_checks(V), cl_qubits(V);
        std::vector<std::vector<int>> cl_frontier_checks(V), cl_frontier_qubits(V);

        auto find_root = [&](auto& self, int x) -> int {
            if (parent[x] == x) return x;
            return parent[x] = self(self, parent[x]);
        };

        auto merge_clusters = [&](int a, int b) -> int {
            a = find_root(find_root, a);
            b = find_root(find_root, b);
            if (a == b) return a;
            if (cl_checks[a].size() + cl_qubits[a].size() < cl_checks[b].size() + cl_qubits[b].size()) {
                std::swap(a, b);
            }
            parent[b] = a;
            cluster_solved[a] = 0;
            cl_checks[a].insert(cl_checks[a].end(), cl_checks[b].begin(), cl_checks[b].end());
            cl_qubits[a].insert(cl_qubits[a].end(), cl_qubits[b].begin(), cl_qubits[b].end());
            cl_frontier_checks[a].insert(cl_frontier_checks[a].end(), cl_frontier_checks[b].begin(), cl_frontier_checks[b].end());
            cl_frontier_qubits[a].insert(cl_frontier_qubits[a].end(), cl_frontier_qubits[b].begin(), cl_frontier_qubits[b].end());
            cl_checks[b].clear(); cl_qubits[b].clear();
            cl_frontier_checks[b].clear(); cl_frontier_qubits[b].clear();
            return a;
        };

        for (int c : active_syns) {
            parent[c] = c;
            cl_checks[c].push_back(c);
            cl_frontier_checks[c].push_back(c);
        }

        std::vector<uint8_t> res_s = s;
        std::vector<int> global_q_to_local(code.N, -1);

        while (true) {
            std::vector<int> unsolved;
            std::unordered_set<int> seen;
            for (int c = 0; c < code.M_check; ++c) {
                if (!res_s[c]) continue;
                int r = find_root(find_root, c);
                if (!cluster_solved[r] && seen.insert(r).second) unsolved.push_back(r);
            }
            if (unsolved.empty()) break;

            for (int r0 : unsolved) {
                int r = find_root(find_root, r0);
                std::vector<int> fc = std::move(cl_frontier_checks[r]);
                cl_frontier_checks[r].clear();
                for (int c : fc) {
                    for (int q : code.checks[c]) {
                        int qnode = code.M_check + q;
                        if (parent[qnode] == -1) {
                            parent[qnode] = r;
                            cl_qubits[r].push_back(q);
                            cl_frontier_qubits[r].push_back(q);
                        } else {
                            r = merge_clusters(r, qnode);
                        }
                    }
                }
            }

            seen.clear();
            std::vector<int> cand_roots;
            for (int c = 0; c < code.M_check; ++c) {
                if (!res_s[c]) continue;
                int r = find_root(find_root, c);
                if (seen.insert(r).second) cand_roots.push_back(r);
            }

            for (int r0 : cand_roots) {
                int r = find_root(find_root, r0);
                std::vector<int> fq = std::move(cl_frontier_qubits[r]);
                cl_frontier_qubits[r].clear();
                for (int q : fq) {
                    for (int c : code.q_to_checks[q]) {
                        if (parent[c] == -1) {
                            parent[c] = r;
                            cl_checks[r].push_back(c);
                            cl_frontier_checks[r].push_back(c);
                        } else {
                            r = merge_clusters(r, c);
                        }
                    }
                }
            }

            seen.clear();
            cand_roots.clear();
            for (int c = 0; c < code.M_check; ++c) {
                if (!res_s[c]) continue;
                int r = find_root(find_root, c);
                if (seen.insert(r).second) cand_roots.push_back(r);
            }

            for (int r : cand_roots) {
                const auto& qs = cl_qubits[r];
                const auto& cs = cl_checks[r];
                if (qs.empty()) continue;
                int nc = (int)qs.size();
                int nr = (int)cs.size();
                for (int j = 0; j < nc; ++j) global_q_to_local[qs[j]] = j;

                std::vector<std::vector<int>> row_cols(nr);
                std::vector<uint8_t> rhs(nr, 0);
                for (int i = 0; i < nr; ++i) {
                    int c_idx = cs[i];
                    rhs[i] = res_s[c_idx];
                    for (int q : code.checks[c_idx]) {
                        int loc = global_q_to_local[q];
                        if (loc != -1) row_cols[i].push_back(loc);
                    }
                }
                for (int j = 0; j < nc; ++j) global_q_to_local[qs[j]] = -1;

                std::vector<uint8_t> local_sol;
                if (solve_gf2_submatrix(nr, nc, row_cols, rhs, local_sol, true)) {
                    for (int j = 0; j < nc; ++j) {
                        if (local_sol[j]) {
                            int q = qs[j];
                            corr[q] ^= 1;
                            for (int c_inc : code.q_to_checks[q]) res_s[c_inc] ^= 1;
                        }
                    }
                    cluster_solved[r] = 1;
                }
            }
        }
        return corr;
    }
};

// ============================================================================
// DECODER 2: Min-Sum Belief Propagation + OSD (BP+OSD)
// ============================================================================
struct BPOSDDecoder {
    const Code& code;
    int max_iter;
    double damping;

    struct Edge {
        int c, q;
        double q_to_c;
        double c_to_q;
    };

    explicit BPOSDDecoder(const Code& c, int iters = 35, double damp = 0.75)
        : code(c), max_iter(iters), damping(damp) {}

    std::vector<uint8_t> decode(const std::vector<uint8_t>& s, double prior_p = 0.02) const {
        std::vector<uint8_t> corr(code.N, 0);
        bool any_syn = false;
        for (uint8_t v : s) if (v) { any_syn = true; break; }
        if (!any_syn) return corr;

        double p_clamped = std::clamp(prior_p, 1e-6, 0.45);
        double l0 = std::log((1.0 - p_clamped) / p_clamped);

        std::vector<std::vector<int>> c_edges(code.M_check), q_edges(code.N);
        std::vector<Edge> edges;
        edges.reserve(code.N * 4);
        for (int c = 0; c < code.M_check; ++c) {
            for (int q : code.checks[c]) {
                int eid = (int)edges.size();
                edges.push_back({c, q, l0, 0.0});
                c_edges[c].push_back(eid);
                q_edges[q].push_back(eid);
            }
        }

        std::vector<double> posterior(code.N, l0);
        for (int it = 0; it < max_iter; ++it) {
            for (int c = 0; c < code.M_check; ++c) {
                const auto& eids = c_edges[c];
                int deg = (int)eids.size();
                if (deg == 0) continue;
                int sign_prod = s[c] ? -1 : 1;
                double min1 = 1e30, min2 = 1e30;
                int min1_idx = -1;
                for (int i = 0; i < deg; ++i) {
                    double v = edges[eids[i]].q_to_c;
                    if (v < 0) sign_prod = -sign_prod;
                    double av = std::abs(v);
                    if (av < min1) { min2 = min1; min1 = av; min1_idx = i; }
                    else if (av < min2) { min2 = av; }
                }
                for (int i = 0; i < deg; ++i) {
                    int sgn = sign_prod * (edges[eids[i]].q_to_c < 0 ? -1 : 1);
                    double mag = 0.85 * (i == min1_idx ? min2 : min1);
                    double new_msg = sgn * mag;
                    edges[eids[i]].c_to_q = damping * new_msg + (1.0 - damping) * edges[eids[i]].c_to_q;
                }
            }

            for (int q = 0; q < code.N; ++q) {
                double sum = l0;
                for (int eid : q_edges[q]) sum += edges[eid].c_to_q;
                posterior[q] = sum;
                corr[q] = (sum < 0.0) ? 1 : 0;
                for (int eid : q_edges[q]) {
                    edges[eid].q_to_c = std::clamp(sum - edges[eid].c_to_q, -30.0, 30.0);
                }
            }

            if (code.verify_syndrome(s, corr)) return corr;
        }

        std::vector<int> order(code.N);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return posterior[a] < posterior[b];
        });

        std::vector<int> inv_order(code.N);
        for (int idx = 0; idx < code.N; ++idx) inv_order[order[idx]] = idx;

        std::vector<std::vector<int>> row_cols(code.M_check);
        for (int c = 0; c < code.M_check; ++c) {
            row_cols[c].reserve(code.checks[c].size());
            for (int q : code.checks[c]) row_cols[c].push_back(inv_order[q]);
        }

        std::vector<uint8_t> perm_sol;
        bool ok = solve_gf2_submatrix(code.M_check, code.N, row_cols, s, perm_sol, true);
        assert(ok);
        for (int idx = 0; idx < code.N; ++idx) corr[order[idx]] = perm_sol[idx];
        return corr;
    }
};

// ============================================================================
// DECODER 3: BDL Hierarchical Unit-Cleaning RG Decoder (rg_strict & rg_relaxed)
// ============================================================================
struct HierRGDecoder {
    const Code& code;
    bool strict_monotonic; // true = BDL proof rule |s_after| <= |s_before|; false = allow 1-rib <= 2x local push

    HierRGDecoder(const Code& c, bool strict_rule)
        : code(c), strict_monotonic(strict_rule) {}

    std::vector<uint8_t> decode(const std::vector<uint8_t>& s) const {
        std::vector<uint8_t> corr(code.N, 0);
        std::vector<uint8_t> res_s = s;

        if (code.rg.valid) {
            std::vector<uint8_t> in_unit(code.M_check, 0);
            std::vector<int> q_loc(code.N, -1);

            for (size_t lv = 0; lv < code.rg.levels.size(); ++lv) {
                for (const auto& unit_checks : code.rg.levels[lv]) {
                    int w_int = 0;
                    for (int c : unit_checks) {
                        in_unit[c] = 1;
                        w_int += res_s[c];
                    }
                    if (w_int == 0) {
                        for (int c : unit_checks) in_unit[c] = 0;
                        continue;
                    }

                    // Collect qubits touching unit_checks; sort strictly-interior qubits FIRST
                    // so Gaussian elimination / OSD uses interior qubits before pushing syndrome outside!
                    std::vector<std::pair<int, int>> cand_q; // (ext_deg, q)
                    std::unordered_set<int> seen_q;
                    for (int c : unit_checks) {
                        for (int q : code.checks[c]) {
                            if (seen_q.insert(q).second) {
                                int ext_deg = 0;
                                for (int c2 : code.q_to_checks[q]) {
                                    if (!in_unit[c2]) ++ext_deg;
                                }
                                cand_q.push_back({ext_deg, q});
                            }
                        }
                    }
                    std::sort(cand_q.begin(), cand_q.end());
                    int nc = (int)cand_q.size();
                    int nr = (int)unit_checks.size();
                    for (int j = 0; j < nc; ++j) q_loc[cand_q[j].second] = j;

                    std::vector<std::vector<int>> row_cols(nr);
                    std::vector<uint8_t> rhs(nr, 0);
                    for (int i = 0; i < nr; ++i) {
                        int c = unit_checks[i];
                        rhs[i] = res_s[c];
                        for (int q : code.checks[c]) row_cols[i].push_back(q_loc[q]);
                    }
                    for (int j = 0; j < nc; ++j) q_loc[cand_q[j].second] = -1;

                    std::vector<uint8_t> sol;
                    if (solve_gf2_submatrix(nr, nc, row_cols, rhs, sol, true)) {
                        // Compute net syndrome weight change globally if we apply `sol`!
                        std::unordered_map<int, uint8_t> toggled;
                        for (int j = 0; j < nc; ++j) {
                            if (!sol[j]) continue;
                            int q = cand_q[j].second;
                            for (int c2 : code.q_to_checks[q]) toggled[c2] ^= 1;
                        }
                        int delta_syn = 0;
                        int ext_added = 0;
                        for (const auto& [c2, fl] : toggled) {
                            if (!fl) continue;
                            int d = res_s[c2] ? -1 : +1;
                            delta_syn += d;
                            if (!in_unit[c2] && d > 0) ++ext_added;
                        }
                        bool accept = strict_monotonic ? (delta_syn <= 0) : (ext_added <= 2 * w_int);
                        if (accept) {
                            for (int j = 0; j < nc; ++j) {
                                if (!sol[j]) continue;
                                int q = cand_q[j].second;
                                corr[q] ^= 1;
                                for (int c2 : code.q_to_checks[q]) res_s[c2] ^= 1;
                            }
                        }
                    }

                    for (int c : unit_checks) in_unit[c] = 0;
                }
            }
        }

        // Top-level base code / any remaining uncleaned residual syndrome -> finish with ClusterUF
        bool rem = false;
        for (uint8_t v : res_s) if (v) { rem = true; break; }
        if (rem) {
            ClusterUFDecoder uf(code);
            auto rem_corr = uf.decode(res_s);
            for (int q = 0; q < code.N; ++q) corr[q] ^= rem_corr[q];
        }
        return corr;
    }
};

std::vector<uint8_t> dispatch_decode(
    const Code& code,
    const std::string& dec_name,
    const std::vector<uint8_t>& s,
    double prior_p
) {
    if (dec_name == "bposd") {
        BPOSDDecoder dec(code, 35, 0.75);
        return dec.decode(s, prior_p);
    } else if (dec_name == "rg_strict") {
        HierRGDecoder dec(code, true);
        return dec.decode(s);
    } else if (dec_name == "rg_relaxed") {
        HierRGDecoder dec(code, false);
        return dec.decode(s);
    } else {
        ClusterUFDecoder dec(code);
        return dec.decode(s);
    }
}

// ============================================================================
// MODE 1: Distance & Energy Barrier
// ============================================================================
int estimate_distance_isd(const Code& code, int trials = 400, uint64_t seed = 42) {
    if (code.K == 0) return -1;
    std::mt19937_64 rng(seed);
    std::vector<int> order(code.N), inv(code.N);
    std::iota(order.begin(), order.end(), 0);

    int best_d = code.N + 1;
    for (int t = 0; t < trials; ++t) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int i = 0; i < code.N; ++i) inv[order[i]] = i;

        int nr = code.M_check, nc = code.N;
        std::vector<BitRow> mat(nr, BitRow(nc));
        for (int r = 0; r < nr; ++r)
            for (int q : code.checks[r]) mat[r].flip(inv[q]);

        std::vector<int> pcol(nr, -1), prow(nc, -1);
        int rank = 0;
        for (int c = 0; c < nc && rank < nr; ++c) {
            int sel = -1;
            for (int r = rank; r < nr; ++r) if (mat[r].get(c)) { sel = r; break; }
            if (sel == -1) continue;
            if (sel != rank) std::swap(mat[sel], mat[rank]);
            pcol[rank] = c; prow[c] = rank;
            for (int r = 0; r < nr; ++r) if (r != rank && mat[r].get(c)) mat[r].xor_with(mat[rank]);
            ++rank;
        }

        std::vector<uint8_t> zero_corr(code.N, 0), vec(code.N, 0);
        for (int c = 0; c < nc; ++c) {
            if (prow[c] != -1) continue;
            std::fill(vec.begin(), vec.end(), 0);
            int w = 1;
            vec[order[c]] = 1;
            for (int r = 0; r < rank; ++r) {
                if (mat[r].get(c)) { vec[order[pcol[r]]] = 1; ++w; }
            }
            if (w < best_d && code.has_logical_error(vec, zero_corr)) {
                bool imp = true;
                while (imp) {
                    imp = false;
                    for (const auto& st : code.stabs) {
                        int delta = 0;
                        for (int q : st) delta += vec[q] ? -1 : +1;
                        if (delta < 0) {
                            for (int q : st) vec[q] ^= 1;
                            w += delta;
                            imp = true;
                        }
                    }
                }
                if (w < best_d) best_d = w;
            }
        }
    }
    return best_d;
}

int estimate_energy_barrier(const Code& code) {
    if (code.K == 0) return -1;
    if (code.N <= 22 && code.M_check <= 60) {
        uint32_t total = 1U << code.N;
        std::vector<uint8_t> dist(total, 255);
        std::vector<uint64_t> syn(total, 0ULL), qmask(code.N, 0ULL);
        for (int c = 0; c < code.M_check; ++c)
            for (int q : code.checks[c]) qmask[q] |= (1ULL << c);
        std::vector<uint32_t> lmask(code.K, 0U);
        for (int a = 0; a < code.K; ++a)
            for (int q : code.logicals[a]) lmask[a] |= (1U << q);

        std::vector<std::vector<uint32_t>> buckets(code.M_check + 1);
        dist[0] = 0;
        buckets[0].push_back(0);
        for (int b = 0; b <= code.M_check; ++b) {
            for (size_t idx = 0; idx < buckets[b].size(); ++idx) {
                uint32_t u = buckets[b][idx];
                if (dist[u] < b) continue;
                if (syn[u] == 0ULL && u != 0U) {
                    for (uint32_t lm : lmask) if (__builtin_popcount(u & lm) & 1) return b;
                }
                for (int q = 0; q < code.N; ++q) {
                    uint32_t v = u ^ (1U << q);
                    uint64_t sv = syn[u] ^ qmask[q];
                    int nb = std::max(b, __builtin_popcountll(sv));
                    if (nb < dist[v]) {
                        dist[v] = (uint8_t)nb;
                        syn[v] = sv;
                        buckets[nb].push_back(v);
                    }
                }
            }
        }
    }

    std::mt19937_64 rng(1337);
    std::vector<int> order(code.N), inv(code.N);
    std::iota(order.begin(), order.end(), 0);
    std::vector<uint8_t> best_log(code.N, 0), zero_corr(code.N, 0);
    int best_w = code.N + 1;

    for (int t = 0; t < 150; ++t) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int i = 0; i < code.N; ++i) inv[order[i]] = i;
        std::vector<BitRow> mat(code.M_check, BitRow(code.N));
        for (int r = 0; r < code.M_check; ++r)
            for (int q : code.checks[r]) mat[r].flip(inv[q]);
        std::vector<int> pcol(code.M_check, -1), prow(code.N, -1);
        int rank = 0;
        for (int c = 0; c < code.N && rank < code.M_check; ++c) {
            int sel = -1;
            for (int r = rank; r < code.M_check; ++r) if (mat[r].get(c)) { sel = r; break; }
            if (sel == -1) continue;
            std::swap(mat[sel], mat[rank]);
            pcol[rank] = c; prow[c] = rank;
            for (int r = 0; r < code.M_check; ++r) if (r != rank && mat[r].get(c)) mat[r].xor_with(mat[rank]);
            ++rank;
        }
        for (int c = 0; c < code.N; ++c) {
            if (prow[c] != -1) continue;
            std::vector<uint8_t> vec(code.N, 0);
            vec[order[c]] = 1;
            int w = 1;
            for (int r = 0; r < rank; ++r) if (mat[r].get(c)) { vec[order[pcol[r]]] = 1; ++w; }
            if (w < best_w && code.has_logical_error(vec, zero_corr)) {
                best_w = w;
                best_log = vec;
            }
        }
    }

    int best_barrier = code.M_check;
    for (int rep = 0; rep < 60; ++rep) {
        std::vector<uint8_t> target = best_log;
        if (rep > 0 && !code.stabs.empty()) {
            int nflips = rng() % 6;
            for (int f = 0; f < nflips; ++f) {
                const auto& st = code.stabs[rng() % code.stabs.size()];
                for (int q : st) target[q] ^= 1;
            }
        }
        std::vector<int> supp;
        for (int q = 0; q < code.N; ++q) if (target[q]) supp.push_back(q);
        for (int tr = 0; tr < 25; ++tr) {
            std::vector<uint8_t> cur_s(code.M_check, 0), used(supp.size(), 0);
            int cur_sw = 0, peak = 0;
            for (size_t step = 0; step < supp.size(); ++step) {
                int best_idx = -1, best_next_sw = 1e9;
                for (size_t i = 0; i < supp.size(); ++i) {
                    if (used[i]) continue;
                    int q = supp[i], dsw = 0;
                    for (int c : code.q_to_checks[q]) dsw += cur_s[c] ? -1 : +1;
                    int cand = cur_sw + dsw;
                    if (cand < best_next_sw || (cand == best_next_sw && (rng() & 1))) {
                        best_next_sw = cand; best_idx = (int)i;
                    }
                }
                used[best_idx] = 1;
                int q = supp[best_idx];
                for (int c : code.q_to_checks[q]) { cur_s[c] ^= 1; cur_sw += cur_s[c] ? +1 : -1; }
                peak = std::max(peak, cur_sw);
                if (peak >= best_barrier) break;
            }
            best_barrier = std::min(best_barrier, peak);
        }
    }
    return best_barrier;
}

// ============================================================================
// MODE 3: Rejection-Free Continuous-Time n-Fold Way Kinetic Monte Carlo (BKL)
// ============================================================================
struct KMCResult {
    double mean_t = 0.0;
    double sem_t = 0.0;
    double median_t = 0.0;
    double mean_flips = 0.0;
    int num_capped = 0;
};

KMCResult run_kmc_parallel(
    const Code& code,
    double beta,
    const std::string& decoder_name,
    int num_trials,
    int64_t max_flips_per_trial,
    int num_threads
) {
    int max_deg = code.max_q_deg;
    int num_buckets = 2 * max_deg + 1;
    std::vector<double> rate_of_bucket(num_buckets, 0.0);
    for (int b = 0; b < num_buckets; ++b) {
        int dE = b - max_deg;
        rate_of_bucket[b] = 1.0 / (1.0 + std::exp(beta * (double)dE));
    }
    double prior_p = 1.0 / (1.0 + std::exp(2.0 * beta));

    std::vector<double> t_fails(num_trials, 0.0);
    std::vector<double> flip_counts(num_trials, 0.0);
    std::atomic<int> next_trial{0};
    std::atomic<int> capped_count{0};

    auto worker = [&](int tid) {
        std::mt19937_64 rng(1000003ULL * (tid + 1) + (uint64_t)(beta * 1000.0) + code.N * 17ULL);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_dist(1.0);

        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0);
        std::vector<int> q_bucket(code.N, 0), q_pos(code.N, 0);
        std::vector<std::vector<int>> buckets(num_buckets);

        auto move_qubit = [&](int q, int new_b) {
            int old_b = q_bucket[q];
            if (old_b == new_b) return;
            int pos = q_pos[q];
            int last_q = buckets[old_b].back();
            buckets[old_b][pos] = last_q;
            q_pos[last_q] = pos;
            buckets[old_b].pop_back();
            q_bucket[q] = new_b;
            q_pos[q] = (int)buckets[new_b].size();
            buckets[new_b].push_back(q);
        };

        while (true) {
            int tr = next_trial.fetch_add(1);
            if (tr >= num_trials) break;

            std::fill(e.begin(), e.end(), 0);
            std::fill(s.begin(), s.end(), 0);
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < code.N; ++q) {
                int b = (int)code.q_to_checks[q].size() + max_deg;
                q_bucket[q] = b;
                q_pos[q] = (int)buckets[b].size();
                buckets[b].push_back(q);
            }

            double t = 0.0, t_next_dec = 0.0;
            int64_t flips = 0;
            bool failed = false;

            while (flips < max_flips_per_trial) {
                double Q_tot = 0.0, Q_cum[64];
                for (int b = 0; b < num_buckets; ++b) {
                    Q_tot += rate_of_bucket[b] * (double)buckets[b].size();
                    Q_cum[b] = Q_tot;
                }
                double dt = exp_dist(rng) / Q_tot;
                t += dt;
                t_next_dec -= dt;

                double r = uni(rng) * Q_tot;
                int cb = 0;
                while (cb < num_buckets - 1 && r >= Q_cum[cb]) ++cb;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < num_buckets - 1) ++cb;

                int q = buckets[cb][rng() % buckets[cb].size()];
                e[q] ^= 1;
                ++flips;

                for (int c : code.q_to_checks[q]) {
                    uint8_t was = s[c];
                    s[c] ^= 1;
                    int db = was ? +2 : -2;
                    for (int q2 : code.checks[c]) move_qubit(q2, q_bucket[q2] + db);
                }

                if (t_next_dec <= 0.0) {
                    auto corr = dispatch_decode(code, decoder_name, s, prior_p);
                    if (code.has_logical_error(e, corr)) {
                        failed = true;
                        break;
                    }
                    t_next_dec = 0.1 * t;
                }
            }
            if (!failed) capped_count.fetch_add(1);
            t_fails[tr] = t;
            flip_counts[tr] = (double)flips;
        }
    };

    int nt = std::max(1, std::min(num_threads, num_trials));
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    KMCResult res;
    double sum = 0.0, sum_fl = 0.0;
    for (int i = 0; i < num_trials; ++i) { sum += t_fails[i]; sum_fl += flip_counts[i]; }
    res.mean_t = sum / num_trials;
    res.mean_flips = sum_fl / num_trials;
    res.num_capped = capped_count.load();
    double var = 0.0;
    for (int i = 0; i < num_trials; ++i) { double d = t_fails[i] - res.mean_t; var += d * d; }
    res.sem_t = (num_trials > 1) ? std::sqrt(var / (num_trials * (num_trials - 1))) : 0.0;
    std::sort(t_fails.begin(), t_fails.end());
    res.median_t = t_fails[num_trials / 2];
    return res;
}

// ============================================================================
// MODE 2: i.i.d. Threshold P_L(p)
// ============================================================================
void run_threshold_mode(const Code& code, double p, const std::string& dec_name, int num_trials, int num_threads) {
    std::atomic<int> next_tr{0}, n_fail{0};
    auto worker = [&](int tid) {
        std::mt19937_64 rng(999983ULL * (tid + 1) + (uint64_t)(p * 1e6));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0);
        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::fill(s.begin(), s.end(), 0);
            for (int q = 0; q < code.N; ++q) {
                e[q] = (uni(rng) < p) ? 1 : 0;
                if (e[q]) for (int c : code.q_to_checks[q]) s[c] ^= 1;
            }
            auto corr = dispatch_decode(code, dec_name, s, p);
            if (code.has_logical_error(e, corr)) n_fail.fetch_add(1);
        }
    };
    int nt = std::max(1, std::min(num_threads, num_trials));
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();
    double pl = (double)n_fail.load() / num_trials;
    double se = std::sqrt(pl * (1.0 - pl) / num_trials);
    std::cout << "THRESHOLD p=" << std::fixed << std::setprecision(4) << p
              << " dec=" << dec_name << " N=" << code.N
              << " trials=" << num_trials << " P_L=" << std::setprecision(5) << pl
              << " sem=" << se << "\n";
}

// ============================================================================
// MODE 4: Passive Thermal Cooling / Initialization from noisy state (BDL Outlook #2)
// ============================================================================
void run_cooling_mode(const Code& code, double beta, double p_init, int num_trials, int64_t max_flips, int num_threads) {
    int max_deg = code.max_q_deg;
    int num_buckets = 2 * max_deg + 1;
    std::vector<double> rate_of_bucket(num_buckets, 0.0);
    for (int b = 0; b < num_buckets; ++b) {
        rate_of_bucket[b] = 1.0 / (1.0 + std::exp(beta * (double)(b - max_deg)));
    }
    std::vector<double> t_grid;
    for (int k = -4; k <= 26; ++k) t_grid.push_back(std::pow(2.0, (double)k));
    int G = (int)t_grid.size();

    std::atomic<int> next_tr{0};
    int nt = std::max(1, std::min(num_threads, num_trials));
    std::vector<std::vector<double>> thread_sums(nt, std::vector<double>(G, 0.0));

    auto worker = [&](int tid) {
        std::mt19937_64 rng(77777ULL * (tid + 1) + (uint64_t)(beta * 100.0));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_dist(1.0);
        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0);
        std::vector<int> q_bucket(code.N, 0), q_pos(code.N, 0);
        std::vector<std::vector<int>> buckets(num_buckets);

        auto move_qubit = [&](int q, int new_b) {
            int old_b = q_bucket[q];
            if (old_b == new_b) return;
            int pos = q_pos[q], last_q = buckets[old_b].back();
            buckets[old_b][pos] = last_q;
            q_pos[last_q] = pos;
            buckets[old_b].pop_back();
            q_bucket[q] = new_b;
            q_pos[q] = (int)buckets[new_b].size();
            buckets[new_b].push_back(q);
        };

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::fill(s.begin(), s.end(), 0);
            for (int q = 0; q < code.N; ++q) {
                e[q] = (uni(rng) < p_init) ? 1 : 0;
                if (e[q]) for (int c : code.q_to_checks[q]) s[c] ^= 1;
            }
            int cur_sw = 0;
            for (uint8_t v : s) cur_sw += v;
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < code.N; ++q) {
                int dE = 0;
                for (int c : code.q_to_checks[q]) dE += s[c] ? -1 : +1;
                int b = dE + max_deg;
                q_bucket[q] = b;
                q_pos[q] = (int)buckets[b].size();
                buckets[b].push_back(q);
            }
            double t = 0.0;
            int g_idx = 0;
            for (int64_t fl = 0; fl < max_flips && g_idx < G; ++fl) {
                double Q_tot = 0.0, Q_cum[64];
                for (int b = 0; b < num_buckets; ++b) {
                    Q_tot += rate_of_bucket[b] * (double)buckets[b].size();
                    Q_cum[b] = Q_tot;
                }
                double dt = exp_dist(rng) / Q_tot;
                while (g_idx < G && t + dt >= t_grid[g_idx]) {
                    thread_sums[tid][g_idx] += (double)cur_sw / code.M_check;
                    ++g_idx;
                }
                if (g_idx >= G) break;
                t += dt;
                double r = uni(rng) * Q_tot;
                int cb = 0;
                while (cb < num_buckets - 1 && r >= Q_cum[cb]) ++cb;
                while (buckets[cb].empty() && cb > 0) --cb;
                while (buckets[cb].empty() && cb < num_buckets - 1) ++cb;
                int q = buckets[cb][rng() % buckets[cb].size()];
                e[q] ^= 1;
                for (int c : code.q_to_checks[q]) {
                    uint8_t was = s[c];
                    s[c] ^= 1;
                    cur_sw += was ? -1 : +1;
                    int db = was ? +2 : -2;
                    for (int q2 : code.checks[c]) move_qubit(q2, q_bucket[q2] + db);
                }
            }
            while (g_idx < G) {
                thread_sums[tid][g_idx] += (double)cur_sw / code.M_check;
                ++g_idx;
            }
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();
    for (int g = 0; g < G; ++g) {
        double tot = 0.0;
        for (int i = 0; i < nt; ++i) tot += thread_sums[i][g];
        std::cout << "COOL beta=" << std::fixed << std::setprecision(2) << beta
                  << " t=" << std::scientific << std::setprecision(3) << t_grid[g]
                  << " syn_density=" << std::fixed << std::setprecision(6) << (tot / num_trials) << "\n";
    }
}

// ============================================================================
// MODE 5: Hybrid Lazy Active QEC + Passive Thermal Bath
// Evolve continuously under Glauber bath at beta; every interval `tau_ec`, measure syndrome
// (with optional syndrome measurement flip error `p_meas` = 0 or >0; if 0, apply exact decoder recovery
//  `e ^= corr`, resetting excitations; if `p_meas > 0`, noisy syndrome -> residual errors stay!)
// Run for total physical duration `T_total = num_cycles * tau_ec` and report survival probability!
// ============================================================================
void run_hybrid_mode(
    const Code& code,
    double beta,
    double tau_ec,
    int num_cycles,
    const std::string& dec_name,
    int num_trials,
    int num_threads
) {
    int max_deg = code.max_q_deg;
    int num_buckets = 2 * max_deg + 1;
    std::vector<double> rate_of_bucket(num_buckets, 0.0);
    for (int b = 0; b < num_buckets; ++b) {
        rate_of_bucket[b] = 1.0 / (1.0 + std::exp(beta * (double)(b - max_deg)));
    }
    double prior_p = 1.0 / (1.0 + std::exp(2.0 * beta));

    std::atomic<int> next_tr{0}, n_fail{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int tid) {
        std::mt19937_64 rng(555557ULL * (tid + 1) + (uint64_t)(tau_ec * 100.0));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_dist(1.0);
        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0), zero_corr(code.N, 0);
        std::vector<int> q_bucket(code.N, 0), q_pos(code.N, 0);
        std::vector<std::vector<int>> buckets(num_buckets);

        auto rebuild_buckets = [&]() {
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < code.N; ++q) {
                int dE = 0;
                for (int c : code.q_to_checks[q]) dE += s[c] ? -1 : +1;
                int b = dE + max_deg;
                q_bucket[q] = b;
                q_pos[q] = (int)buckets[b].size();
                buckets[b].push_back(q);
            }
        };

        auto move_qubit = [&](int q, int new_b) {
            int old_b = q_bucket[q];
            if (old_b == new_b) return;
            int pos = q_pos[q], last_q = buckets[old_b].back();
            buckets[old_b][pos] = last_q;
            q_pos[last_q] = pos;
            buckets[old_b].pop_back();
            q_bucket[q] = new_b;
            q_pos[q] = (int)buckets[new_b].size();
            buckets[new_b].push_back(q);
        };

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::fill(e.begin(), e.end(), 0);
            std::fill(s.begin(), s.end(), 0);
            rebuild_buckets();

            bool failed = false;
            for (int cyc = 0; cyc < num_cycles; ++cyc) {
                double elapsed = 0.0;
                while (elapsed < tau_ec) {
                    double Q_tot = 0.0, Q_cum[64];
                    for (int b = 0; b < num_buckets; ++b) {
                        Q_tot += rate_of_bucket[b] * (double)buckets[b].size();
                        Q_cum[b] = Q_tot;
                    }
                    double dt = exp_dist(rng) / Q_tot;
                    if (elapsed + dt >= tau_ec) break;
                    elapsed += dt;

                    double r = uni(rng) * Q_tot;
                    int cb = 0;
                    while (cb < num_buckets - 1 && r >= Q_cum[cb]) ++cb;
                    while (buckets[cb].empty() && cb > 0) --cb;
                    while (buckets[cb].empty() && cb < num_buckets - 1) ++cb;
                    int q = buckets[cb][rng() % buckets[cb].size()];
                    e[q] ^= 1;
                    for (int c : code.q_to_checks[q]) {
                        uint8_t was = s[c];
                        s[c] ^= 1;
                        int db = was ? +2 : -2;
                        for (int q2 : code.checks[c]) move_qubit(q2, q_bucket[q2] + db);
                    }
                }
                // Active QEC correction cycle!
                auto corr = dispatch_decode(code, dec_name, s, prior_p);
                for (int q = 0; q < code.N; ++q) e[q] ^= corr[q];
                std::fill(s.begin(), s.end(), 0);
                if (code.has_logical_error(e, zero_corr)) {
                    failed = true;
                    break;
                }
                rebuild_buckets();
            }
            if (failed) n_fail.fetch_add(1);
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();
    double pl = (double)n_fail.load() / num_trials;
    std::cout << "HYBRID beta=" << std::fixed << std::setprecision(2) << beta
              << " tau_ec=" << std::scientific << std::setprecision(3) << tau_ec
              << " cycles=" << num_cycles
              << " T_tot=" << (tau_ec * num_cycles)
              << " N=" << code.N
              << " P_fail=" << std::fixed << std::setprecision(5) << pl << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: kmc_engine <stats|threshold|kmc|cool|hybrid> <code.txt> ...\n";
        return 1;
    }
    std::string mode = argv[1];
    Code code = Code::load(argv[2]);
    int hw_threads = (int)std::thread::hardware_concurrency();
    if (hw_threads <= 0) hw_threads = 4;

    if (mode == "stats") {
        int d = estimate_distance_isd(code, (code.N <= 600) ? 400 : 120);
        int eb = estimate_energy_barrier(code);
        std::cout << "STATS N=" << code.N << " M_check=" << code.M_check
                  << " K=" << code.K << " max_q_deg=" << code.max_q_deg
                  << " rg=" << (code.rg.valid ? (int)code.rg.levels.size() : 0)
                  << " d<=" << d << " Delta<=" << eb << "\n";
    } else if (mode == "threshold") {
        run_threshold_mode(code, std::stod(argv[3]), argv[4], std::stoi(argv[5]),
                           (argc >= 7) ? std::stoi(argv[6]) : hw_threads);
    } else if (mode == "kmc") {
        auto res = run_kmc_parallel(code, std::stod(argv[3]), argv[4], std::stoi(argv[5]),
                                    std::stoll(argv[6]), (argc >= 8) ? std::stoi(argv[7]) : hw_threads);
        std::cout << "KMC beta=" << std::fixed << std::setprecision(2) << std::stod(argv[3])
                  << " dec=" << argv[4] << " N=" << code.N << " trials=" << argv[5]
                  << " mean_t=" << std::scientific << std::setprecision(4) << res.mean_t
                  << " sem_t=" << res.sem_t << " median_t=" << res.median_t
                  << " mean_flips=" << std::fixed << std::setprecision(1) << res.mean_flips
                  << " capped=" << res.num_capped << "\n";
    } else if (mode == "cool") {
        run_cooling_mode(code, std::stod(argv[3]), std::stod(argv[4]), std::stoi(argv[5]),
                         std::stoll(argv[6]), (argc >= 8) ? std::stoi(argv[7]) : hw_threads);
    } else if (mode == "hybrid") {
        run_hybrid_mode(code, std::stod(argv[3]), std::stod(argv[4]), std::stoi(argv[5]),
                        argv[6], std::stoi(argv[7]), (argc >= 9) ? std::stoi(argv[8]) : hw_threads);
    }
    return 0;
}
