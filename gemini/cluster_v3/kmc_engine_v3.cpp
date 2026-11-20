// kmc_engine_v3.cpp
// V3 Next-Generation Engine for Ultra-Compact 3D-Local & Non-Local Quantum Memories (MIT 6.S980).
// Modes:
//   1. prep_logicals  : Fast 64-bit BitRow GF(2) dual-logical computation
//   2. stats          : Exact rank/meta-checks M_meta, check/qubit degrees, ISD distance d, & barrier Delta
//   3. kmc            : Continuous-time BKL Glauber KMC supporting non-uniform graded couplings J_c (in 0.1 steps!)
//   4. anneal         : Cooling / glassy-pinning test supporting graded energy funnels J_c!
//   5. mfqec          : Gate-Based Digital Measurement-Free QEC (Hong-Guo-Lucas Nat. Comm. 2025) with gate noise p_gate
//   6. singleshot     : Multi-round Single-Shot Active QEC under noisy syndrome extraction (p_data, q_meas)
//                       via Soft-BP + Augmented [H | I_M] Cluster OSD/MAP decoding!

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
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
    std::vector<int> j_tenths; // 10 * J_c (default 10 => J_c = 1.0)

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
            if (jin) {
                for (int i = 0; i < c.M_check; ++i) jin >> c.j_tenths[i];
            }
        }
        return c;
    }

    bool has_logical_error(const std::vector<uint8_t>& e, const std::vector<uint8_t>& corr) const {
        for (int a = 0; a < K; ++a) {
            uint8_t p = 0;
            for (int q : logicals[a]) p ^= (e[q] ^ corr[q]);
            if (p) return true;
        }
        return false;
    }
};

void compute_and_append_dual_logicals(const std::string& raw_path, const std::string& out_path) {
    std::ifstream in(raw_path);
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
            if (has_spiv[p]) x.xor_with(stab_piv[p]);
            else { stab_piv[p] = std::move(x); has_spiv[p] = 1; spiv_list.push_back(p); break; }
        }
    }
    std::sort(spiv_list.rbegin(), spiv_list.rend());
    for (int p : spiv_list) {
        for (int p2 : spiv_list) if (p2 != p && stab_piv[p2].get(p)) stab_piv[p2].xor_with(stab_piv[p]);
    }
    std::vector<BitRow> quot_piv(N);
    std::vector<uint8_t> has_qpiv(N, 0);
    for (const auto& r : checks) {
        BitRow x(N);
        for (int c : r) x.flip(c);
        while (true) {
            int p = x.first_one();
            if (p == -1) break;
            if (has_qpiv[p]) x.xor_with(quot_piv[p]);
            else { quot_piv[p] = std::move(x); has_qpiv[p] = 1; break; }
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
            if (has_qpiv[p]) x.xor_with(quot_piv[p]);
            else {
                quot_piv[p] = std::move(x); has_qpiv[p] = 1;
                std::vector<int> supp;
                for (int q = 0; q < N; ++q) if (v.get(q)) supp.push_back(q);
                dual_logs.push_back(std::move(supp));
                break;
            }
        }
    }
    std::ofstream out(out_path);
    out << N << " " << M_check << " " << M_stab << " " << dual_logs.size() << "\n";
    for (const auto& r : checks) { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
    for (const auto& r : stabs)  { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
    for (const auto& r : dual_logs) { out << r.size(); for (int q : r) out << " " << q; out << "\n"; }
}

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
    for (int r = rank; r < nr; ++r) if (mat[r].get(nc)) return false;
    sol_out.assign(nc, 0);
    for (int r = 0; r < rank; ++r) if (mat[r].get(nc)) sol_out[pcol[r]] = 1;

    if (min_weight_refine && nc > rank && nc <= 600) {
        BitRow cur(nc);
        for (int c = 0; c < nc; ++c) if (sol_out[c]) cur.flip(c);
        int cur_w = cur.popcount();
        std::vector<BitRow> nvs;
        for (int c = 0; c < nc && (int)nvs.size() < 48; ++c) {
            if (prow[c] != -1) continue;
            BitRow nv(nc); nv.flip(c);
            for (int r = 0; r < rank; ++r) if (mat[r].get(c)) nv.flip(pcol[r]);
            nvs.push_back(std::move(nv));
        }
        bool imp = true; int passes = 0;
        while (imp && passes < 4) {
            imp = false; ++passes;
            for (const auto& nv : nvs) {
                int w2 = 0;
                for (size_t i = 0; i < cur.w.size(); ++i) w2 += __builtin_popcountll(cur.w[i] ^ nv.w[i]);
                if (w2 < cur_w) { cur.xor_with(nv); cur_w = w2; imp = true; }
            }
        }
        for (int c = 0; c < nc; ++c) sol_out[c] = cur.get(c) ? 1 : 0;
    }
    return true;
}

struct FastClusterUF {
    const Code& code;
    int V;
    mutable std::vector<int> parent, global_q_to_local, touched_nodes, seen_stamp;
    mutable std::vector<uint8_t> cluster_solved, res_s;
    mutable std::vector<std::vector<int>> cl_checks, cl_qubits, cl_f_checks, cl_f_qubits;
    mutable int cur_stamp = 1;

    explicit FastClusterUF(const Code& c)
        : code(c), V(c.M_check + c.N), parent(V, -1), global_q_to_local(c.N, -1),
          seen_stamp(V, 0), cluster_solved(V, 0), res_s(c.M_check, 0),
          cl_checks(V), cl_qubits(V), cl_f_checks(V), cl_f_qubits(V) {}

    inline int find_root(int x) const {
        int r = x;
        while (parent[r] != r) r = parent[r];
        while (x != r) { int nxt = parent[x]; parent[x] = r; x = nxt; }
        return r;
    }
    inline void touch(int node) const { if (parent[node] == -1) touched_nodes.push_back(node); }

    int merge_clusters(int a, int b) const {
        a = find_root(a); b = find_root(b);
        if (a == b) return a;
        if (cl_checks[a].size() + cl_qubits[a].size() < cl_checks[b].size() + cl_qubits[b].size()) std::swap(a, b);
        parent[b] = a; cluster_solved[a] = 0;
        cl_checks[a].insert(cl_checks[a].end(), cl_checks[b].begin(), cl_checks[b].end());
        cl_qubits[a].insert(cl_qubits[a].end(), cl_qubits[b].begin(), cl_qubits[b].end());
        cl_f_checks[a].insert(cl_f_checks[a].end(), cl_f_checks[b].begin(), cl_f_checks[b].end());
        cl_f_qubits[a].insert(cl_f_qubits[a].end(), cl_f_qubits[b].begin(), cl_f_qubits[b].end());
        cl_checks[b].clear(); cl_qubits[b].clear(); cl_f_checks[b].clear(); cl_f_qubits[b].clear();
        return a;
    }

    void decode_into(const std::vector<int>& active_checks, const std::vector<uint8_t>& s_in,
                     std::vector<uint8_t>& corr_out, const std::vector<double>* q_llr = nullptr) const {
        std::fill(corr_out.begin(), corr_out.end(), 0);
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
                        } else r = merge_clusters(r, qnode);
                    }
                }
            }
            ++cur_stamp; cand_roots.clear();
            for (int c : active_checks) {
                if (!res_s[c]) continue;
                int r = find_root(c);
                if (seen_stamp[r] != cur_stamp) { seen_stamp[r] = cur_stamp; cand_roots.push_back(r); }
            }
            for (int r0 : cand_roots) {
                int r = find_root(r0);
                std::vector<int> fq = std::move(cl_f_qubits[r]); cl_f_qubits[r].clear();
                for (int q : fq) {
                    for (int c : code.q_to_checks[q]) {
                        if (parent[c] == -1) {
                            touch(c); parent[c] = r; res_s[c] = s_in[c];
                            cl_checks[r].push_back(c); cl_f_checks[r].push_back(c);
                        } else r = merge_clusters(r, c);
                    }
                }
            }
            ++cur_stamp; cand_roots.clear();
            for (int c : active_checks) {
                if (!res_s[c]) continue;
                int r = find_root(c);
                if (seen_stamp[r] != cur_stamp) { seen_stamp[r] = cur_stamp; cand_roots.push_back(r); }
            }
            for (int r : cand_roots) {
                auto& qs = cl_qubits[r];
                const auto& cs = cl_checks[r];
                if (qs.empty()) continue;
                if (q_llr != nullptr) {
                    std::stable_sort(qs.begin(), qs.end(), [&](int a, int b) { return (*q_llr)[a] < (*q_llr)[b]; });
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
                            int q = qs[j]; corr_out[q] ^= 1;
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
};

// ============================================================================
// SINGLE-SHOT AUGMENTED BP + LOCAL OSD DECODER ON [H | I_M]
// Decodes noisy syndrome `s_tilde = H e_data ^ e_meas` in ONE measurement round!
// ============================================================================
struct SingleShotDecoder {
    const Code& code;
    struct Edge { int c, q; double q_to_c, c_to_q; };
    std::vector<std::vector<int>> c_edges, q_edges;
    mutable std::vector<Edge> edges;
    mutable std::vector<double> post_q, post_m;

    explicit SingleShotDecoder(const Code& c)
        : code(c), c_edges(c.M_check), q_edges(c.N), post_q(c.N, 0.0), post_m(c.M_check, 0.0) {
        for (int ci = 0; ci < code.M_check; ++ci) {
            for (int q : code.checks[ci]) {
                int eid = (int)edges.size();
                edges.push_back({ci, q, 0.0, 0.0});
                c_edges[ci].push_back(eid);
                q_edges[q].push_back(eid);
            }
        }
    }

    void decode_single_shot(const std::vector<uint8_t>& s_tilde, double p_data, double q_meas,
                            std::vector<uint8_t>& data_corr_out) const {
        std::fill(data_corr_out.begin(), data_corr_out.end(), 0);
        bool any_s = false;
        for (uint8_t v : s_tilde) if (v) { any_s = true; break; }
        if (!any_s) return;

        double pd = std::clamp(p_data, 1e-6, 0.45);
        double qm = std::clamp(q_meas, 1e-6, 0.45);
        double l_data = std::log((1.0 - pd) / pd);
        double l_meas = std::log((1.0 - qm) / qm);
        const double damp = 0.78;

        for (auto& e : edges) { e.q_to_c = l_data; e.c_to_q = 0.0; }

        // 22 iterations of damped Min-Sum BP on [H | I_M]
        // Note: on check c, variable m_c has degree 1, so its message to check c is ALWAYS +l_meas!
        for (int it = 0; it < 22; ++it) {
            for (int c = 0; c < code.M_check; ++c) {
                const auto& eids = c_edges[c];
                int deg = (int)eids.size();
                int sign_prod = s_tilde[c] ? -1 : 1; // m_c -> c message has + sign (l_meas > 0)
                double min1 = l_meas, min2 = 1e30;
                int min1_idx = -2; // -2 denotes the measurement variable m_c!
                for (int i = 0; i < deg; ++i) {
                    double v = edges[eids[i]].q_to_c;
                    if (v < 0.0) sign_prod = -sign_prod;
                    double av = std::abs(v);
                    if (av < min1) { min2 = min1; min1 = av; min1_idx = i; }
                    else if (av < min2) { min2 = av; }
                }
                // Update c -> data_q messages
                for (int i = 0; i < deg; ++i) {
                    int sgn = sign_prod * (edges[eids[i]].q_to_c < 0.0 ? -1 : 1);
                    double mag = 0.85 * (i == min1_idx ? min2 : min1);
                    double nm = sgn * mag;
                    edges[eids[i]].c_to_q = damp * nm + (1.0 - damp) * edges[eids[i]].c_to_q;
                }
                // Posterior of m_c: l_meas + (c -> m_c)
                double c_to_m = sign_prod * 0.85 * (min1_idx == -2 ? min2 : min1);
                post_m[c] = l_meas + c_to_m;
            }
            for (int q = 0; q < code.N; ++q) {
                double sum = l_data;
                for (int eid : q_edges[q]) sum += edges[eid].c_to_q;
                post_q[q] = sum;
                data_corr_out[q] = (sum < 0.0) ? 1 : 0;
                for (int eid : q_edges[q]) {
                    edges[eid].q_to_c = std::clamp(sum - edges[eid].c_to_q, -30.0, 30.0);
                }
            }
        }

        // Post-BP local greedy LLR refinement on residual unsatisfied checks where BP prefers data flip over meas flip!
        std::vector<uint8_t> res_s = s_tilde;
        for (int q = 0; q < code.N; ++q) {
            if (data_corr_out[q]) for (int c : code.q_to_checks[q]) res_s[c] ^= 1;
        }
        // Any qubit q whose flip strictly lowers total weighted NLL cost `l_data * |e_d| + l_meas * |s_tilde ^ H e_d|` is flipped!
        std::vector<int> q_ord(code.N);
        std::iota(q_ord.begin(), q_ord.end(), 0);
        std::sort(q_ord.begin(), q_ord.end(), [&](int a, int b) { return post_q[a] < post_q[b]; });
        bool improved = true;
        int passes = 0;
        while (improved && passes < 6) {
            improved = false; ++passes;
            for (int q : q_ord) {
                double delta_cost = data_corr_out[q] ? -l_data : +l_data;
                for (int c : code.q_to_checks[q]) {
                    delta_cost += res_s[c] ? -l_meas : +l_meas;
                }
                if (delta_cost < -1e-9) {
                    data_corr_out[q] ^= 1;
                    for (int c : code.q_to_checks[q]) res_s[c] ^= 1;
                    improved = true;
                }
            }
        }
    }
};

// ============================================================================
// STATS MODE (including exact meta-check count M_meta!)
// ============================================================================
void run_stats_v3(const Code& code) {
    // 1. Exact GF(2) rank of H_check -> M_meta = M_check - rank
    std::vector<BitRow> piv(code.N);
    std::vector<uint8_t> has_p(code.N, 0);
    int rk = 0;
    for (const auto& r : code.checks) {
        BitRow x(code.N);
        for (int q : r) x.flip(q);
        while (true) {
            int p = x.first_one();
            if (p == -1) break;
            if (has_p[p]) x.xor_with(piv[p]);
            else { piv[p] = std::move(x); has_p[p] = 1; ++rk; break; }
        }
    }
    int m_meta = code.M_check - rk;

    // 2. ISD distance & greedy barrier upper bound
    std::mt19937_64 rng(42);
    std::vector<int> order(code.N), inv(code.N);
    std::iota(order.begin(), order.end(), 0);
    std::vector<uint8_t> best_log(code.N, 0), zero_c(code.N, 0);
    int best_d = code.N + 1;
    int trials = (code.N <= 1200) ? 280 : 90;

    for (int t = 0; t < trials && code.K > 0; ++t) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int i = 0; i < code.N; ++i) inv[order[i]] = i;
        std::vector<BitRow> mat(code.M_check, BitRow(code.N));
        for (int r = 0; r < code.M_check; ++r) for (int q : code.checks[r]) mat[r].flip(inv[q]);
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
        std::vector<uint8_t> vec(code.N, 0);
        for (int c = 0; c < code.N; ++c) {
            if (prow[c] != -1) continue;
            std::fill(vec.begin(), vec.end(), 0);
            vec[order[c]] = 1; int w = 1;
            for (int r = 0; r < rank; ++r) if (mat[r].get(c)) { vec[order[pcol[r]]] = 1; ++w; }
            if (w < best_d && code.has_logical_error(vec, zero_c)) {
                best_d = w; best_log = vec;
            }
        }
    }

    int best_bar = (code.K > 0) ? code.M_check : -1;
    for (int rep = 0; rep < 45 && code.K > 0; ++rep) {
        std::vector<uint8_t> target = best_log;
        if (rep > 0 && !code.stabs.empty()) {
            int nf = rng() % 5;
            for (int f = 0; f < nf; ++f) for (int q : code.stabs[rng() % code.stabs.size()]) target[q] ^= 1;
        }
        std::vector<int> supp;
        for (int q = 0; q < code.N; ++q) if (target[q]) supp.push_back(q);
        for (int tr = 0; tr < 20; ++tr) {
            std::vector<uint8_t> cs(code.M_check, 0), used(supp.size(), 0);
            int cur_sw = 0, peak = 0;
            for (size_t st = 0; st < supp.size(); ++st) {
                int bi = -1, bnext = 1e9;
                for (size_t i = 0; i < supp.size(); ++i) {
                    if (used[i]) continue;
                    int dsw = 0;
                    for (int c : code.q_to_checks[supp[i]]) dsw += cs[c] ? -1 : +1;
                    if (cur_sw + dsw < bnext || (cur_sw + dsw == bnext && (rng() & 1))) {
                        bnext = cur_sw + dsw; bi = (int)i;
                    }
                }
                used[bi] = 1;
                for (int c : code.q_to_checks[supp[bi]]) { cs[c] ^= 1; cur_sw += cs[c] ? +1 : -1; }
                peak = std::max(peak, cur_sw);
                if (peak >= best_bar) break;
            }
            best_bar = std::min(best_bar, peak);
        }
    }
    int max_cw = 0; double sum_cw = 0.0;
    for (const auto& r : code.checks) { max_cw = std::max(max_cw, (int)r.size()); sum_cw += r.size(); }
    std::cout << "STATS N=" << code.N << " M=" << code.M_check << " M_meta=" << m_meta
              << " K=" << code.K << " max_cw=" << max_cw
              << " mean_cw=" << std::fixed << std::setprecision(2) << (code.M_check ? sum_cw / code.M_check : 0.0)
              << " qdeg=" << code.max_q_deg << " d<=" << best_d << " Delta<=" << best_bar << "\n";
}

// ============================================================================
// WEIGHTED BKL KMC (`kmc` mode — supports non-uniform graded couplings J_c!)
// ============================================================================
void run_kmc_weighted(const Code& code, double beta, int num_trials, int64_t max_flips, int num_threads) {
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
    std::atomic<int> next_tr{0}, capped{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int tid) {
        FastClusterUF uf(code);
        std::mt19937_64 rng(1234577ULL * (tid + 1) + (uint64_t)(beta * 1000) + code.N * 19ULL);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::exponential_distribution<double> exp_d(1.0);

        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0), corr(code.N, 0);
        std::vector<int> q_b(code.N, 0), q_p(code.N, 0), act, pos_act(code.M_check, -1);
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
            std::fill(e.begin(), e.end(), 0);
            for (int c : act) { s[c] = 0; pos_act[c] = -1; } act.clear();
            for (auto& bk : buckets) bk.clear();
            for (int q = 0; q < code.N; ++q) {
                int dE10 = 0;
                for (int c : code.q_to_checks[q]) dE10 += code.j_tenths[c];
                int b = dE10 + max_dE10;
                q_b[q] = b; q_p[q] = (int)buckets[b].size(); buckets[b].push_back(q);
            }
            double t = 0.0, t_next = 1.0;
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
                e[q] ^= 1; ++flips;
                for (int c : code.q_to_checks[q]) {
                    uint8_t was = s[c];
                    tog_c(c);
                    int db = (was ? +2 : -2) * code.j_tenths[c];
                    for (int q2 : code.checks[c]) move_q(q2, q_b[q2] + db);
                }
                if (t_next <= 0.0) {
                    uf.decode_into(act, s, corr);
                    if (code.has_logical_error(e, corr)) { failed = true; break; }
                    t_next = 0.15 * t;
                }
            }
            if (!failed) capped.fetch_add(1);
            t_fails[tr] = t;
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    double sum = 0.0; for (double v : t_fails) sum += v;
    double mean_t = sum / num_trials;
    std::sort(t_fails.begin(), t_fails.end());
    std::cout << "KMC beta=" << std::fixed << std::setprecision(2) << beta
              << " N=" << code.N << " K=" << code.K << " trials=" << num_trials
              << " mean_t=" << std::scientific << std::setprecision(4) << mean_t
              << " median_t=" << t_fails[num_trials / 2] << " capped=" << capped.load() << "\n";
}

// ============================================================================
// GATE-BASED DIGITAL MEASUREMENT-FREE QEC (`mfqec` — Hong-Guo-Lucas Nat Comm 2025!)
// Discrete sweeps of local coherent Metropolis cooling + ancilla resets with gate noise `p_gate`,
// fighting storage error `p_env` per sweep WITHOUT mid-circuit readout/decoding!
// ============================================================================
void run_mfqec_digital(
    const Code& code,
    double beta_alg,
    double p_env,
    double p_gate,
    int num_sweeps,
    int num_trials,
    int num_threads
) {
    std::atomic<int> next_tr{0}, n_fail_uf{0}, n_fail_raw{0};
    int nt = std::max(1, std::min(num_threads, num_trials));
    std::vector<double> syn_sum(nt, 0.0);

    auto worker = [&](int tid) {
        FastClusterUF uf(code);
        std::mt19937_64 rng(918273645ULL * (tid + 1) + (uint64_t)(p_gate * 1e7) + (uint64_t)(p_env * 1e7));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0), corr(code.N, 0), zero_c(code.N, 0);
        std::vector<int> q_perm(code.N);
        std::iota(q_perm.begin(), q_perm.end(), 0);

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::fill(e.begin(), e.end(), 0);
            std::fill(s.begin(), s.end(), 0);

            for (int sw = 0; sw < num_sweeps; ++sw) {
                // 1. Environmental idling error before cooling sweep
                if (p_env > 0.0) {
                    for (int q = 0; q < code.N; ++q) {
                        if (uni(rng) < p_env) {
                            e[q] ^= 1;
                            for (int c : code.q_to_checks[q]) s[c] ^= 1;
                        }
                    }
                }
                // 2. Digital local cooling sweep (coherent local check-ancilla interaction + ancilla reset)
                std::shuffle(q_perm.begin(), q_perm.end(), rng);
                for (int q : q_perm) {
                    int deg = (int)code.q_to_checks[q].size();
                    if (deg == 0) continue;
                    // Coherent check-evaluation gate noise: each CNOT touching q can fault or misread ancilla!
                    int noisy_dE10 = 0;
                    for (int c : code.q_to_checks[q]) {
                        uint8_t sc_meas = s[c];
                        if (p_gate > 0.0 && uni(rng) < p_gate) sc_meas ^= 1; // noisy coherent ancilla comparison
                        noisy_dE10 += (sc_meas ? -1 : +1) * code.j_tenths[c];
                    }
                    double dE = 0.1 * (double)noisy_dE10;
                    // Metropolis-Gibbs dissipative jump acceptance: flip if dE < 0, wait if dE == 0 (or small thermal rate)!
                    double p_flip = (dE < 0.0) ? 1.0 : std::exp(-beta_alg * std::max(dE, 0.5));
                    if (uni(rng) < p_flip) {
                        e[q] ^= 1;
                        for (int c : code.q_to_checks[q]) s[c] ^= 1;
                    }
                    // Gate backaction on data qubit from the local cooling gadget (proportional to deg * p_gate * 0.25)
                    if (p_gate > 0.0 && uni(rng) < 0.25 * deg * p_gate) {
                        e[q] ^= 1;
                        for (int c : code.q_to_checks[q]) s[c] ^= 1;
                    }
                }
            }

            int sw_cnt = 0;
            std::vector<int> act;
            for (int c = 0; c < code.M_check; ++c) if (s[c]) { ++sw_cnt; act.push_back(c); }
            syn_sum[tid] += (double)sw_cnt / code.M_check;
            if (code.has_logical_error(e, zero_c)) n_fail_raw.fetch_add(1);
            uf.decode_into(act, s, corr);
            if (code.has_logical_error(e, corr)) n_fail_uf.fetch_add(1);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    double tot_syn = 0.0; for (double v : syn_sum) tot_syn += v;
    std::cout << "MFQEC sweeps=" << num_sweeps << " p_env=" << std::fixed << std::setprecision(4) << p_env
              << " p_gate=" << std::setprecision(5) << p_gate
              << " N=" << code.N << " K=" << code.K
              << " P_fail=" << std::setprecision(4) << ((double)n_fail_uf.load() / num_trials)
              << " P_raw=" << ((double)n_fail_raw.load() / num_trials)
              << " end_syn=" << std::setprecision(5) << (tot_syn / num_trials) << "\n";
}

// ============================================================================
// MULTI-ROUND SINGLE-SHOT ACTIVE QEC (`singleshot` mode!)
// 1 noisy syndrome extraction per round (`q_meas = p_data`) for `num_rounds` rounds!
// ============================================================================
void run_singleshot_qec(
    const Code& code,
    double p_data,
    double q_meas,
    int num_rounds,
    int num_trials,
    int num_threads
) {
    std::atomic<int> next_tr{0}, n_fail{0};
    int nt = std::max(1, std::min(num_threads, num_trials));

    auto worker = [&](int tid) {
        SingleShotDecoder ss_dec(code);
        FastClusterUF final_uf(code);
        std::mt19937_64 rng(564738291ULL * (tid + 1) + (uint64_t)(p_data * 1e6));
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<uint8_t> e(code.N, 0), s_true(code.M_check, 0), s_tilde(code.M_check, 0), corr(code.N, 0);

        while (true) {
            int tr = next_tr.fetch_add(1);
            if (tr >= num_trials) break;
            std::fill(e.begin(), e.end(), 0);

            for (int r = 0; r < num_rounds; ++r) {
                // 1. Data errors accumulate on top of residual uncorrected error from previous round!
                for (int q = 0; q < code.N; ++q) {
                    if (uni(rng) < p_data) e[q] ^= 1;
                }
                // 2. Single-shot noisy syndrome measurement (ONLY 1 measurement per check, flipped w.p. q_meas!)
                for (int c = 0; c < code.M_check; ++c) {
                    uint8_t p = 0;
                    for (int q : code.checks[c]) p ^= e[q];
                    if (uni(rng) < q_meas) p ^= 1;
                    s_tilde[c] = p;
                }
                // 3. Single-shot recovery on [H | I_M]
                ss_dec.decode_single_shot(s_tilde, p_data, q_meas, corr);
                for (int q = 0; q < code.N; ++q) e[q] ^= corr[q];
            }

            // Final ideal destructive logical readout at end of `num_rounds` single-shot cycles!
            std::vector<int> act;
            for (int c = 0; c < code.M_check; ++c) {
                uint8_t p = 0;
                for (int q : code.checks[c]) p ^= e[q];
                s_true[c] = p;
                if (p) act.push_back(c);
            }
            final_uf.decode_into(act, s_true, corr);
            if (code.has_logical_error(e, corr)) n_fail.fetch_add(1);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker, i);
    for (auto& th : pool) th.join();

    std::cout << "SINGLESHOT rounds=" << num_rounds
              << " p_data=" << std::fixed << std::setprecision(4) << p_data
              << " q_meas=" << q_meas << " N=" << code.N << " K=" << code.K
              << " P_fail=" << std::setprecision(4) << ((double)n_fail.load() / num_trials) << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    std::string mode = argv[1];
    if (mode == "prep_logicals") {
        compute_and_append_dual_logicals(argv[2], argv[3]);
        return 0;
    }
    int hw = std::max(1, (int)std::thread::hardware_concurrency());
    if (mode == "stats") {
        Code c = Code::load(argv[2]);
        run_stats_v3(c);
    } else if (mode == "kmc") {
        // kmc <code_path> <j_path_or_uniform> <beta> <trials> <max_flips> [threads]
        Code c = Code::load(argv[2], argv[3]);
        run_kmc_weighted(c, std::stod(argv[4]), std::stoi(argv[5]), std::stoll(argv[6]),
                         (argc >= 8) ? std::stoi(argv[7]) : hw);
    } else if (mode == "mfqec") {
        // mfqec <code_path> <j_path_or_uniform> <beta_alg> <p_env> <p_gate> <sweeps> <trials> [threads]
        Code c = Code::load(argv[2], argv[3]);
        run_mfqec_digital(c, std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]),
                          std::stoi(argv[7]), std::stoi(argv[8]), (argc >= 10) ? std::stoi(argv[9]) : hw);
    } else if (mode == "singleshot") {
        // singleshot <code_path> <p_data> <q_meas> <rounds> <trials> [threads]
        Code c = Code::load(argv[2]);
        run_singleshot_qec(c, std::stod(argv[3]), std::stod(argv[4]),
                           std::stoi(argv[5]), std::stoi(argv[6]), (argc >= 8) ? std::stoi(argv[7]) : hw);
    }
    return 0;
}
