#include "grouped_gemm.h"
#include "input_generator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const std::string& action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(action + ": " + cudaGetErrorString(status));
    }
}

void usage() {
    std::cout << R"(HPCA Grouped GEMM assignment benchmark
Usage: ./grouped_gemm [options]

  --impl=baseline|v1|v2|v3|v4|v5|bonus    (default: baseline)
  --case=NAME     public preset or custom (see --list-cases)
  --iters=10      measured iterations (default: 10)
  --warmup=3      untimed iterations (default: 3)
  --seed=2026     reproducible data seed (default: 2026)
  --verify        check packed output against CPU reference
  --list-cases    show public presets
  --help          show this help

Custom case only:
  --case=custom --k=17 --n=19 --m-list=0,1,31,2,7
  --case=custom --groups=16 --k=128 --n=256 --min-m=0 --max-m=128
                --pattern=uniform|random|skewed|sparse --seed=2026

The provided baseline is runnable. V1-V5 return NOT IMPLEMENTED until
students fill in src/grouped_gemm_kernels.cu.
)";
}

void print_cases() {
    std::cout << "Public named cases (G / K / N / M_g):\n"
              << "  toy       3 / 2 / 2         [2,1,3] (fixed numeric values)\n"
              << "  tiny      7 / 11 / 13       [0,1,2,5,7,0,9]\n"
              << "  small     64 / 256 / 256    4 + (13*g mod 29)\n"
              << "  balanced  16 / 256 / 256    64 + 8*(g mod 7)\n"
              << "  skewed    32 / 256 / 256    few large, many small or empty\n"
              << "  boundary  13 / 63 / 70      [0,1,2,3,7,15,16,17,31,32,33,63,65]\n"
              << "  wide      12 / 512 / 384    96 + 16*(g mod 5)\n"
              << "  many-tiny 256 / 128 / 128   empty groups and 1..16 rows\n"
              << "  medium    8 / 512 / 512     256 + 32*(g mod 4)\n"
              << "  large     4 / 2048 / 2048   1024 rows per group\n"
              << "  large-skewed 32 / 512 / 512 four 4096-row groups, many small or empty\n"
              << "  empty     4 / 17 / 19       [0,0,0,0]\n"
              << "  custom    user-provided dimensions and sizes/pattern\n";
}

int parse_int(const std::string& text, const std::string& name) {
    size_t end = 0;
    int val = 0;
    try { val = std::stoi(text, &end); }
    catch (const std::exception&) { throw std::invalid_argument("bad " + name); }
    if (end != text.size()) throw std::invalid_argument("bad " + name);
    return val;
}

uint32_t parse_seed(const std::string& text) {
    size_t end = 0;
    unsigned long long val = 0;
    try { val = std::stoull(text, &end); }
    catch (const std::exception&) { throw std::invalid_argument("bad --seed"); }
    if (end != text.size() || val > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("--seed must be an unsigned 32-bit integer");
    }
    return static_cast<uint32_t>(val);
}

struct Args {
    InputSpec spec;
    std::string impl = "baseline";
    int iters = 10;
    int warmup = 3;
    bool verify = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    bool saw_custom_param = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        if (a == "--list-cases") { print_cases(); std::exit(0); }
        if (a == "--verify") { args.verify = true; continue; }
        const size_t eq = a.find('=');
        if (eq == std::string::npos) throw std::invalid_argument("unknown argument: " + a);
        const std::string name = a.substr(0, eq);
        const std::string val = a.substr(eq + 1);
        if (name == "--impl") args.impl = val;
        else if (name == "--case") args.spec.case_name = val;
        else if (name == "--iters") args.iters = parse_int(val, name);
        else if (name == "--warmup") args.warmup = parse_int(val, name);
        else if (name == "--seed") args.spec.seed = parse_seed(val);
        else if (name == "--groups") { args.spec.groups = parse_int(val, name); saw_custom_param = true; }
        else if (name == "--k") { args.spec.k = parse_int(val, name); saw_custom_param = true; }
        else if (name == "--n") { args.spec.n = parse_int(val, name); saw_custom_param = true; }
        else if (name == "--min-m") { args.spec.min_m = parse_int(val, name); saw_custom_param = true; }
        else if (name == "--max-m") { args.spec.max_m = parse_int(val, name); saw_custom_param = true; }
        else if (name == "--pattern") { args.spec.pattern = val; saw_custom_param = true; }
        else if (name == "--m-list") { args.spec.m_list = val; saw_custom_param = true; }
        else throw std::invalid_argument("unknown argument: " + a);
    }
    if (args.iters < 1 || args.iters > 1000 || args.warmup < 0 || args.warmup > 1000) {
        throw std::invalid_argument("--iters must be 1..1000, --warmup must be 0..1000");
    }
    if (saw_custom_param && args.spec.case_name != "custom") {
        throw std::invalid_argument("custom shape options require --case=custom");
    }
    return args;
}

using Launcher = cudaError_t (*)(const GroupedGemmProblem&, cudaStream_t);
Launcher select_launcher(const std::string& impl) {
    if (impl == "baseline") return launch_baseline;
    if (impl == "v1") return launch_v1;
    if (impl == "v2") return launch_v2;
    if (impl == "v3") return launch_v3;
    if (impl == "v4") return launch_v4;
    if (impl == "v5") return launch_v5;
    if (impl == "bonus") return launch_bonus;
    throw std::invalid_argument("unknown --impl: " + impl);
}

struct DeviceStorage {
    float *x = nullptr, *w = nullptr, *bias = nullptr, *z = nullptr, *y = nullptr;
    int* row_offsets = nullptr;
    ~DeviceStorage() {
        if (x) cudaFree(x);
        if (w) cudaFree(w);
        if (bias) cudaFree(bias);
        if (z) cudaFree(z);
        if (y) cudaFree(y);
        if (row_offsets) cudaFree(row_offsets);
    }
    void allocate(const HostProblem& h, bool needs_z) {
        auto allocate_float = [](float** ptr, size_t count) {
            check(cudaMalloc(reinterpret_cast<void**>(ptr), std::max<size_t>(count, 1) * sizeof(float)),
                  "cudaMalloc tensor");
        };
        allocate_float(&x, h.x.size());
        allocate_float(&w, h.w.size());
        allocate_float(&bias, h.bias.size());
        if (needs_z) allocate_float(&z, static_cast<size_t>(h.total_rows()) * h.n);
        allocate_float(&y, static_cast<size_t>(h.total_rows()) * h.n);
        check(cudaMalloc(reinterpret_cast<void**>(&row_offsets),
                         h.row_offsets.size() * sizeof(int)), "cudaMalloc row_offsets");
        if (!h.x.empty()) check(cudaMemcpy(x, h.x.data(), h.x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy X");
        if (!h.w.empty()) check(cudaMemcpy(w, h.w.data(), h.w.size() * sizeof(float), cudaMemcpyHostToDevice), "copy W");
        if (!h.bias.empty()) check(cudaMemcpy(bias, h.bias.data(), h.bias.size() * sizeof(float), cudaMemcpyHostToDevice), "copy bias");
        check(cudaMemcpy(row_offsets, h.row_offsets.data(), h.row_offsets.size() * sizeof(int),
                         cudaMemcpyHostToDevice), "copy offsets");
    }
};

void print_shape(const HostProblem& h, const Args& args) {
    int min_m = std::numeric_limits<int>::max(), max_m = 0, zero_groups = 0;
    for (int g = 0; g < h.groups; ++g) {
        const int m = h.row_offsets[g + 1] - h.row_offsets[g];
        min_m = std::min(min_m, m);
        max_m = std::max(max_m, m);
        if (!m) ++zero_groups;
    }
    std::cout << "CASE " << args.spec.case_name << "  impl=" << args.impl
              << "  G=" << h.groups << "  K=" << h.k << "  N=" << h.n
              << "  rows=" << h.total_rows() << "  M_range=[" << min_m << "," << max_m
              << "]  empty_groups=" << zero_groups << "  seed=" << args.spec.seed << "\n";
    std::cout << "M_g = [";
    for (int g = 0; g < std::min(h.groups, 20); ++g) {
        if (g) std::cout << ",";
        std::cout << h.row_offsets[g + 1] - h.row_offsets[g];
    }
    if (h.groups > 20) std::cout << ",...";
    std::cout << "]\n";
}

void verify_result(const HostProblem& h, const DeviceStorage& d) {
    const auto expected = cpu_reference(h);
    std::vector<float> actual(expected.size());
    if (!actual.empty()) {
        check(cudaMemcpy(actual.data(), d.y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
              "copy Y for verification");
    }
    constexpr float abs_tol = 1e-3f;
    constexpr float rel_tol = 1e-3f;
    size_t bad = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float ref = expected[i];
        const float got = actual[i];
        const float err = std::fabs(got - ref);
        if (std::isfinite(err)) {
            max_abs = std::max(max_abs, err);
            max_rel = std::max(max_rel, err / std::max(std::fabs(ref), 1e-6f));
        }
        if (!std::isfinite(got) || err > abs_tol + rel_tol * std::fabs(ref)) {
            if (bad < 8) {
                std::cerr << "  mismatch output[" << i << "]: got " << got
                          << ", expected " << ref << "\n";
            }
            ++bad;
        }
    }
    if (bad != 0) {
        throw std::runtime_error("FAIL: " + std::to_string(bad) +
                                 " output mismatches (abs_tol=1e-3, rel_tol=1e-3)");
    }
    std::cout << "PASS: CPU reference (atol=1e-3, rtol=1e-3)  max_abs="
              << max_abs << "  max_rel=" << max_rel << "\n";
}

double median(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    const size_t mid = xs.size() / 2;
    return xs.size() % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2.0;
}

void run(const Args& args) {
    const auto launcher = select_launcher(args.impl);
    const HostProblem h = make_host_problem(args.spec);
    print_shape(h, args);

    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 1) throw std::runtime_error("no CUDA GPU available");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
    std::cout << "GPU " << props.name << "  SM " << props.major << "." << props.minor << "\n";

    DeviceStorage d;
    const bool needs_z = args.impl == "baseline" || args.impl == "v1" ||
                         args.impl == "v2" || args.impl == "v3";
    d.allocate(h, needs_z);
    const GroupedGemmProblem p = {
        h.groups, h.k, h.n, h.total_rows(),
        h.row_offsets.data(), d.row_offsets, d.x, d.w, d.bias, d.z, d.y
    };
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");

    try {
        auto launch = [&]() {
            const cudaError_t status = launcher(p, stream);
            if (status == cudaErrorNotSupported) {
                throw std::runtime_error("NOT IMPLEMENTED: " + args.impl +
                    " (edit only src/grouped_gemm_kernels.cu)");
            }
            check(status, "kernel enqueue");
        };
        if (args.verify) {
            check(cudaMemsetAsync(d.y, 0xff,
                  std::max<size_t>(static_cast<size_t>(h.total_rows()) * h.n, 1) * sizeof(float),
                  stream), "poison output before verification");
            launch();
            check(cudaStreamSynchronize(stream), "verification sync");
            verify_result(h, d);
        }

        for (int i = 0; i < args.warmup; ++i) launch();
        check(cudaStreamSynchronize(stream), "warmup sync");

        cudaEvent_t start = nullptr, stop = nullptr;
        check(cudaEventCreate(&start), "cudaEventCreate(start)");
        check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        std::vector<double> gpu_ms, wall_ms;
        try {
            for (int i = 0; i < args.iters; ++i) {
                const auto wall_start = std::chrono::steady_clock::now();
                check(cudaEventRecord(start, stream), "start event");
                launch();
                check(cudaEventRecord(stop, stream), "stop event");
                check(cudaEventSynchronize(stop), "stop event sync");
                const auto wall_stop = std::chrono::steady_clock::now();
                float gpu = 0.0f;
                check(cudaEventElapsedTime(&gpu, start, stop), "cudaEventElapsedTime");
                gpu_ms.push_back(static_cast<double>(gpu));
                wall_ms.push_back(std::chrono::duration<double, std::milli>(wall_stop - wall_start).count());
            }
        } catch (...) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            throw;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        const double med_gpu = median(gpu_ms);
        const double med_wall = median(wall_ms);
        const double flops = 2.0 * h.total_rows() * h.k * h.n;
        const double gflops = med_gpu > 0 ? flops / (med_gpu * 1e6) : 0;
        std::cout << std::fixed << std::setprecision(4)
                  << "Median GPU event time: " << med_gpu << " ms\n"
                  << "Median host wall time: " << med_wall << " ms\n"
                  << "Effective GEMM throughput: " << gflops << " GFLOP/s\n"
                  << "RESULT,impl=" << args.impl << ",case=" << args.spec.case_name
                  << ",G=" << h.groups << ",K=" << h.k << ",N=" << h.n
                  << ",rows=" << h.total_rows() << ",gpu_ms=" << med_gpu
                  << ",wall_ms=" << med_wall << ",gflops=" << gflops << "\n";
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

} // namespace

int main(int argc, char** argv) {
    try {
        run(parse_args(argc, argv));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
