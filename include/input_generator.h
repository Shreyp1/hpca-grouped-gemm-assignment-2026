#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct InputSpec {
    std::string case_name = "toy";
    int groups = 8;           // custom generated case only
    int k = 128;              // custom case only
    int n = 128;              // custom case only
    int min_m = 1;            // custom generated case only
    int max_m = 64;           // custom generated case only
    std::string pattern = "random";  // uniform, random, skewed, sparse
    std::string m_list;       // optional for custom; comma-separated M_g
    uint32_t seed = 2026;    // deterministic content and custom shapes
};

struct HostProblem {
    int groups = 0;
    int k = 0;
    int n = 0;
    std::vector<int> row_offsets;
    std::vector<float> x;
    std::vector<float> w;
    std::vector<float> bias;

    int total_rows() const { return row_offsets.empty() ? 0 : row_offsets.back(); }
};

// Throws std::invalid_argument if the input violates the public contract.
HostProblem make_host_problem(const InputSpec& spec);
std::vector<float> cpu_reference(const HostProblem& input);
