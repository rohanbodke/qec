// eval_coarse_uf.cpp
// Evaluates exact FastDecoderV7 Union-Find logical failure probability on a coarse CSS code A
// under structured local-interface error channels (p_flat, p_junc, p_span), reusing Code and
// FastDecoderV7 verbatim from cluster_v9/kmc_engine_v17.cpp with zero decoder discrepancy.

#define main kmc_v17_unused_main
#include "../cluster_v9/kmc_engine_v17.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: eval_coarse_uf <coarse.code> <coarse.roles>\n";
        return 1;
    }
    Code code = Code::load(argv[1], "uniform");
    std::ifstream rin(argv[2]);
    if (!rin) {
        std::cerr << "Cannot open roles file " << argv[2] << "\n";
        return 1;
    }
    int n_roles = 0;
    rin >> n_roles;
    if (n_roles != code.N) {
        std::cerr << "N mismatch in roles file\n";
        return 1;
    }
    std::vector<int> q_role(code.N, 0);
    for (int q = 0; q < code.N; ++q) rin >> q_role[q];
    int n_pairs = 0;
    rin >> n_pairs;
    std::vector<std::pair<int, int>> span_pairs(n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
        rin >> span_pairs[i].first >> span_pairs[i].second;
    }

    FastDecoderV7 dec(code, false);
    std::vector<uint8_t> e(code.N, 0), s(code.M_check, 0), corr(code.N, 0);
    std::vector<int> dirty_e, act;

    double p_flat = 0.0, p_junc = 0.0, p_span = 0.0;
    int shots = 0;
    uint64_t seed = 0;
    while (std::cin >> p_flat >> p_junc >> p_span >> shots >> seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        int fails = 0;
        for (int sh = 0; sh < shots; ++sh) {
            for (int q : dirty_e) e[q] = 0;
            dirty_e.clear();
            for (int c : act) s[c] = 0;
            act.clear();

            auto flip_q = [&](int q) {
                if (!e[q]) dirty_e.push_back(q);
                e[q] ^= 1;
            };

            if (p_flat > 0.0 || p_junc > 0.0) {
                for (int q = 0; q < code.N; ++q) {
                    double pq = (q_role[q] == 0) ? p_flat : p_junc;
                    if (pq > 0.0 && uni(rng) < pq) flip_q(q);
                }
            }
            if (p_span > 0.0) {
                for (const auto& pr : span_pairs) {
                    if (uni(rng) < p_span) {
                        flip_q(pr.first);
                        flip_q(pr.second);
                    }
                }
            }

            for (int q : dirty_e) {
                if (!e[q]) continue;
                for (int c : code.q_to_checks[q]) s[c] ^= 1;
            }
            for (int c = 0; c < code.M_check; ++c) {
                if (s[c]) act.push_back(c);
            }

            bool err = false;
            if (act.empty()) {
                err = code.has_logical_error_raw(e);
            } else {
                dec.decode_into(act, s, corr);
                err = code.has_logical_error(e, corr);
            }
            if (err) ++fails;
        }
        std::cout << std::fixed << std::setprecision(6) << ((double)fails / (double)shots) << "\n" << std::flush;
    }
    return 0;
}
