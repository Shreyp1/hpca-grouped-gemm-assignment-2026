#include "grouped_gemm.h"

// Provided teaching baseline. It is intentionally not coalesced:
// neighboring lanes compute different output rows of one column.
// This is NOT a solution to V1-V5; students must implement those
// variants independently in grouped_gemm_kernels.cu.
namespace {

__global__ void baseline_gemm_kernel(const float* x, const float* w,
                                     float* z, int row_begin, int m,
                                     int k, int n, int group) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    if (row >= m) return;

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += x[(row_begin + row) * k + kk] *
               w[(group * k + kk) * n + col];
    }
    z[(row_begin + row) * n + col] = acc;
}

__global__ void baseline_pointwise_kernel(const float* z, const float* bias,
                                          float* y, int row_begin, int m,
                                          int n, int group) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    const int col = idx % n;
    const int offset = row_begin * n + idx;
    const float val = z[offset] + bias[group * n + col];
    y[offset] = fmaxf(0.0f, val);
}

int div_up(int a, int b) { return (a + b - 1) / b; }

} // namespace

cudaError_t launch_baseline(const GroupedGemmProblem& p, cudaStream_t stream) {
    for (int g = 0; g < p.groups; ++g) {
        const int begin = p.h_row_offsets[g];
        const int m = p.h_row_offsets[g + 1] - begin;
        if (m == 0) continue;

        // One thread calculates one GEMM output, then a separate
        // pointwise pass writes the result of bias and ReLU.
        baseline_gemm_kernel<<<dim3(div_up(m, 128), p.n), 128, 0, stream>>>(
            p.x, p.w, p.z, begin, m, p.k, p.n, g);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;

        baseline_pointwise_kernel<<<div_up(m * p.n, 256), 256, 0, stream>>>(
            p.z, p.bias, p.y, begin, m, p.n, g);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}
