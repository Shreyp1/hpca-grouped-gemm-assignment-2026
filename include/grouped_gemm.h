#pragma once

#include <cuda_runtime.h>

// Stable interface. Do not change this header; graders use the same declarations.
// All matrix buffers live on the GPU, in flat, contiguous, row-major storage.
struct GroupedGemmProblem {
    int groups;                 // G >= 1
    int k;                      // common reduction dimension, K
    int n;                      // common output dimension, N
    int total_rows;             // sum_g (M_g)

    const int* h_row_offsets;   // host metadata, length G+1
    const int* d_row_offsets;   // device metadata, length G+1
    const float* x;            // device, [total_rows, K]
    const float* w;            // device, [G, K, N]
    const float* bias;         // device, [G, N]
    float* z;                  // scratch, [total_rows, N]; null for V4/V5/bonus
    float* y;                  // device output, [total_rows, N]
};

// One launcher per assignment variant, defined in src/grouped_gemm_kernels.cu.
// Launch asynchronous work into the supplied stream; never synchronize it.
// Return cudaSuccess on successful enqueue, otherwise a CUDA error code.
cudaError_t launch_v1(const GroupedGemmProblem& p, cudaStream_t stream);
cudaError_t launch_v2(const GroupedGemmProblem& p, cudaStream_t stream);
cudaError_t launch_v3(const GroupedGemmProblem& p, cudaStream_t stream);
cudaError_t launch_v4(const GroupedGemmProblem& p, cudaStream_t stream);
cudaError_t launch_v5(const GroupedGemmProblem& p, cudaStream_t stream);
cudaError_t launch_bonus(const GroupedGemmProblem& p, cudaStream_t stream);

// Provided, intentionally unoptimized baseline. Do not edit src/baseline.cu.
cudaError_t launch_baseline(const GroupedGemmProblem& p, cudaStream_t stream);
