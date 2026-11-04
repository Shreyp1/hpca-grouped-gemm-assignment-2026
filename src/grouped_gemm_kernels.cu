#include "grouped_gemm.h"

// ================================================================
// HPCA 2026: THIS IS THE ONLY FILE STUDENTS MODIFY AND SUBMIT.
// ================================================================
//
// Implement all five variants below. You may add kernels, device
// functions, constants, and helper functions IN THIS FILE.
//
// The provided driver allocates and initializes all buffers.
// V1-V3: write GEMM results to p.z; apply bias/ReLU in a separate
//        kernel to produce p.y. (One pair of launches per nonempty group.)
// V4:    fuse the epilogue; write p.y without an intermediate GEMM
//        result; launch one fused kernel per nonempty group.
// V5:    retain V4's fused computation; process all nonempty groups
//        with exactly one CUDA kernel launch.
// Bonus: optional compact scheduling, also with one launch.
//
// p.h_row_offsets is HOST memory. Device kernels must use
// p.d_row_offsets, or host-derived scalar kernel arguments.
// Never dereference p.h_row_offsets on the GPU.
//
// The TODO return code is deliberate: unimplemented variants must
// fail visibly, not silently produce incorrect or placeholder output.

cudaError_t launch_v1(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // TODO
}

cudaError_t launch_v2(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // TODO
}

cudaError_t launch_v3(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // TODO
}

cudaError_t launch_v4(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // TODO
}

cudaError_t launch_v5(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // TODO
}

cudaError_t launch_bonus(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // OPTIONAL
}
