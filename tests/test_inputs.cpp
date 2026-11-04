#include "input_generator.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void test_case(const std::string& name) {
    InputSpec spec;
    spec.case_name = name;
    const auto p = make_host_problem(spec);
    assert(p.groups >= 1);
    assert(p.row_offsets.size() == static_cast<size_t>(p.groups + 1));
    assert(p.row_offsets.front() == 0);
    assert(p.total_rows() >= 0);
    for (int g = 0; g < p.groups; ++g) assert(p.row_offsets[g] <= p.row_offsets[g+1]);
    assert(p.x.size() == static_cast<size_t>(p.total_rows()) * p.k);
    assert(p.w.size() == static_cast<size_t>(p.groups) * p.k * p.n);
    assert(p.bias.size() == static_cast<size_t>(p.groups) * p.n);
    const auto ref = cpu_reference(p);
    assert(ref.size() == static_cast<size_t>(p.total_rows()) * p.n);
    for (float v : ref) assert(std::isfinite(v) && v >= 0.0f);
    const auto again = make_host_problem(spec);
    assert(p.row_offsets == again.row_offsets);
    assert(p.x == again.x && p.w == again.w && p.bias == again.bias);
}

static void expect_invalid(const InputSpec& spec) {
    bool rejected = false;
    try {
        (void)make_host_problem(spec);
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

int main() {
    for (const char* name : {"toy", "tiny", "small", "balanced", "skewed", "boundary", "wide", "empty"}) {
        test_case(name);
    }
    InputSpec toy_spec;
    toy_spec.case_name = "toy";
    const auto toy = make_host_problem(toy_spec);
    assert((toy.row_offsets == std::vector<int>{0, 2, 3, 6}));
    assert((cpu_reference(toy) == std::vector<float>{0, 2, 3, 5, 2, 0, 5, 0, 6, 0, 0, 1}));

    InputSpec custom;
    custom.case_name = "custom";
    custom.k = 17;
    custom.n = 19;
    custom.m_list = "0,1,31,2,0,7";
    const auto explicit_sizes = make_host_problem(custom);
    assert((explicit_sizes.row_offsets == std::vector<int>{0, 0, 1, 32, 34, 34, 41}));
    for (const char* pattern : {"random", "uniform", "skewed", "sparse"}) {
        custom.m_list.clear();
        custom.pattern = pattern;
        custom.groups = 24;
        custom.min_m = 0;
        custom.max_m = 77;
        const auto p = make_host_problem(custom);
        assert(p.groups == 24);
        assert(p.row_offsets.front() == 0);
        assert(p.total_rows() <= 24 * 77);
    }

    for (int groups : {129, 256, 257}) {
        for (bool explicit_list : {false, true}) {
            InputSpec limit;
            limit.case_name = "custom";
            limit.groups = groups;
            limit.k = 1;
            limit.n = 1;
            limit.min_m = 1;
            limit.max_m = 1;
            limit.pattern = "uniform";
            if (explicit_list) {
                limit.m_list = "1";
                for (int g = 1; g < groups; ++g) limit.m_list += ",1";
            }
            bool accepted = false;
            try {
                const auto p = make_host_problem(limit);
                assert(p.groups == groups);
                assert(p.row_offsets.size() == static_cast<size_t>(groups + 1));
                assert(p.total_rows() == groups);
                accepted = true;
            } catch (const std::invalid_argument&) {}
            assert(accepted == (groups <= 256));
        }
    }

    InputSpec bounds;
    bounds.case_name = "custom";
    bounds.groups = 1;
    bounds.k = 4096;
    bounds.n = 4096;
    bounds.min_m = 0;
    bounds.max_m = 4096;
    bounds.pattern = "uniform";
    {
        const auto p = make_host_problem(bounds);
        assert(p.k == 4096 && p.n == 4096 && p.total_rows() == 4096);
        assert(p.x.size() == 4096u * 4096u);
        assert(p.w.size() == 4096u * 4096u);
    }

    bounds.k = 4097;
    expect_invalid(bounds);
    bounds.k = 4096;
    bounds.n = 4097;
    expect_invalid(bounds);
    bounds.n = 4096;
    bounds.max_m = 4097;
    expect_invalid(bounds);
    bounds.max_m = 4096;
    bounds.m_list = "4097";
    expect_invalid(bounds);
    bounds.m_list.clear();

    bounds.k = 1;
    bounds.n = 1;
    bounds.groups = 16;
    assert(make_host_problem(bounds).total_rows() == 65536);
    bounds.m_list = "4096";
    for (int g = 1; g < 16; ++g) bounds.m_list += ",4096";
    assert(make_host_problem(bounds).total_rows() == 65536);
    bounds.m_list += ",1";
    expect_invalid(bounds);
    bounds.m_list.clear();
    bounds.groups = 17;
    expect_invalid(bounds);

    bounds.groups = 16;
    bounds.k = 4096;
    bounds.n = 4096;
    bounds.max_m = 0;
    expect_invalid(bounds);

    bool rejected = false;
    try {
        custom.m_list = "1,-1,3";
        (void)make_host_problem(custom);
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    rejected = false;
    try {
        custom.m_list = "1,2,";
        (void)make_host_problem(custom);
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    std::cout << "PASS: input contract, fixed toy example, all presets, and determinism\n";
}
