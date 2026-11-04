#include "input_generator.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Rng {
    explicit Rng(uint32_t seed) : state(seed == 0 ? 0x9e3779b9u : seed) {}
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    uint32_t state;
};

int draw_int(Rng& rng, int low, int high) {
    if (low > high) throw std::invalid_argument("invalid random integer range");
    return low + static_cast<int>(rng.next() % static_cast<uint32_t>(high - low + 1));
}

std::vector<int> parse_m_list(const std::string& s) {
    std::vector<int> sizes;
    std::stringstream input(s);
    std::string token;
    while (std::getline(input, token, ',')) {
        if (token.empty()) throw std::invalid_argument("empty M value in --m-list");
        size_t consumed = 0;
        int m = 0;
        try {
            m = std::stoi(token, &consumed);
        } catch (const std::exception&) {
            throw std::invalid_argument("noninteger M value in --m-list");
        }
        if (consumed != token.size()) throw std::invalid_argument("invalid M value in --m-list");
        sizes.push_back(m);
    }
    if (s.empty() || s.back() == ',' || sizes.empty()) {
        throw std::invalid_argument("--m-list must contain at least one integer and no trailing comma");
    }
    return sizes;
}

void validate(int groups, int k, int n, const std::vector<int>& sizes) {
    if (groups < 1 || groups > 256) throw std::invalid_argument("G must be in [1, 256]");
    if (k < 1 || k > 4096 || n < 1 || n > 4096) {
        throw std::invalid_argument("K and N must be in [1, 4096]");
    }
    if (static_cast<int>(sizes.size()) != groups) throw std::invalid_argument("M list/G mismatch");

    int64_t rows = 0;
    for (int m : sizes) {
        if (m < 0 || m > 4096) throw std::invalid_argument("each M_g must be in [0, 4096]");
        rows += m;
    }
    if (rows > 65536) throw std::invalid_argument("total rows must be <= 65536");

    const int64_t elems = rows * k + static_cast<int64_t>(groups) * k * n +
                          static_cast<int64_t>(groups) * n + 2 * rows * n;
    const int64_t bytes = elems * static_cast<int64_t>(sizeof(float)) +
                          (static_cast<int64_t>(groups) + 1) * sizeof(int);
    if (bytes > 1024LL * 1024 * 1024) {
        throw std::invalid_argument("total device tensor storage exceeds 1 GiB");
    }
}

} // namespace

HostProblem make_host_problem(const InputSpec& spec) {
    HostProblem out;
    std::vector<int> sizes;

    if (spec.case_name == "toy") {
        out.k = 2;
        out.n = 2;
        sizes = {2, 1, 3};
    } else if (spec.case_name == "tiny") {
        out.k = 11;
        out.n = 13;
        sizes = {0, 1, 2, 5, 7, 0, 9};
    } else if (spec.case_name == "small") {
        out.k = 256;
        out.n = 256;
        for (int g = 0; g < 64; ++g) sizes.push_back(4 + (g * 13) % 29);
    } else if (spec.case_name == "balanced") {
        out.k = 256;
        out.n = 256;
        for (int g = 0; g < 16; ++g) sizes.push_back(64 + 8 * (g % 7));
    } else if (spec.case_name == "skewed") {
        out.k = 256;
        out.n = 256;
        for (int g = 0; g < 32; ++g) {
            sizes.push_back(g % 7 == 0 ? 256 + 32 * (g % 3) :
                            (g % 5 == 0 ? 0 : 4 * (1 + g % 4)));
        }
    } else if (spec.case_name == "boundary") {
        out.k = 63;
        out.n = 70;
        sizes = {0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 65};
    } else if (spec.case_name == "wide") {
        out.k = 512;
        out.n = 384;
        for (int g = 0; g < 12; ++g) sizes.push_back(96 + 16 * (g % 5));
    } else if (spec.case_name == "many-tiny") {
        out.k = 128;
        out.n = 128;
        for (int g = 0; g < 256; ++g) {
            sizes.push_back(g % 5 == 0 ? 0 : 1 + (7 * g) % 16);
        }
    } else if (spec.case_name == "medium") {
        out.k = 512;
        out.n = 512;
        for (int g = 0; g < 8; ++g) sizes.push_back(256 + 32 * (g % 4));
    } else if (spec.case_name == "large") {
        out.k = 2048;
        out.n = 2048;
        sizes.assign(4, 1024);
    } else if (spec.case_name == "large-skewed") {
        out.k = 512;
        out.n = 512;
        for (int g = 0; g < 32; ++g) {
            sizes.push_back(g % 8 == 0 ? 4096 : (g % 5 == 0 ? 0 : 1 + g % 16));
        }
    } else if (spec.case_name == "empty") {
        out.k = 17;
        out.n = 19;
        sizes = {0, 0, 0, 0};
    } else if (spec.case_name == "custom") {
        out.k = spec.k;
        out.n = spec.n;
        if (!spec.m_list.empty()) {
            sizes = parse_m_list(spec.m_list);
        } else {
            if (spec.groups < 1 || spec.groups > 256) {
                throw std::invalid_argument("custom G must be in [1, 256]");
            }
            if (spec.min_m < 0 || spec.max_m < spec.min_m || spec.max_m > 4096) {
                throw std::invalid_argument("custom M range must satisfy 0 <= min-m <= max-m <= 4096");
            }
            Rng shape_rng(spec.seed ^ 0x6d2b79f5u);
            for (int g = 0; g < spec.groups; ++g) {
                int m = 0;
                if (spec.pattern == "uniform") {
                    m = spec.max_m;
                } else if (spec.pattern == "random") {
                    m = draw_int(shape_rng, spec.min_m, spec.max_m);
                } else if (spec.pattern == "skewed") {
                    const int small_hi = spec.min_m + (spec.max_m - spec.min_m) / 8;
                    m = (g % 8 == 0) ? spec.max_m :
                        ((g % 5 == 0) ? 0 : draw_int(shape_rng, spec.min_m, small_hi));
                } else if (spec.pattern == "sparse") {
                    m = (g % 3 == 0) ? 0 : draw_int(shape_rng, spec.min_m, spec.max_m);
                } else {
                    throw std::invalid_argument("invalid pattern; use uniform, random, skewed, sparse");
                }
                sizes.push_back(m);
            }
        }
    } else {
        throw std::invalid_argument("unknown --case: " + spec.case_name);
    }

    out.groups = static_cast<int>(sizes.size());
    validate(out.groups, out.k, out.n, sizes);
    out.row_offsets.reserve(out.groups + 1);
    out.row_offsets.push_back(0);
    for (int m : sizes) out.row_offsets.push_back(out.row_offsets.back() + m);

    out.x.resize(static_cast<size_t>(out.total_rows()) * out.k);
    out.w.resize(static_cast<size_t>(out.groups) * out.k * out.n);
    out.bias.resize(static_cast<size_t>(out.groups) * out.n);

    if (spec.case_name == "toy") {
        out.x = {1, 2, 3, 1, 2, -1, -1, 2, 0, 3, 2, 1};
        out.w = {1, 2, -1, 1, 2, -1, 1, 3, -1, 1, 2, -2};
        out.bias = {1, -2, -1, 2, 0, 1};
    } else {
        Rng value_rng(spec.seed ^ 0xa341316cu);
        auto fill = [&value_rng](std::vector<float>& values) {
            for (float& v : values) {
                // Exactly representable values in [-1, 1], including zeros.
                const int q = static_cast<int>(value_rng.next() % 129u) - 64;
                v = static_cast<float>(q) / 64.0f;
            }
        };
        fill(out.x);
        fill(out.w);
        fill(out.bias);
    }
    return out;
}
