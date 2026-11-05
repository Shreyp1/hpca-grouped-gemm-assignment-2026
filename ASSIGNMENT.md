# HPCA 2026 GPU Assignment - Requirements

## Scope

Optimize the constrained grouped GEMM followed by bias addition and ReLU:

$$
Y_g[i,j] = \max\left(0,\sum_{k=0}^{K-1} X_g[i,k]W_g[k,j] + b_g[j]\right).
$$

The values of $K$ and $N$ are common to all groups; $M_g$ may differ. All matrices are FP32 and contiguous row-major. The driver supplies packed buffers, row offsets, correctness checking, a baseline, and timing. See [README.md](README.md) for the motivation and visual example and [INPUT_CONTRACT.md](docs/INPUT_CONTRACT.md) for exact bounds and layouts.

## General requirements

1. Implement **all five required variants**, preserving the performance idea of each previous variant in its successor. The bonus is optional.
2. Edit **only** `src/grouped_gemm_kernels.cu`. You may add kernels and helpers in this file. Leave all public function signatures and provided files unchanged. The submission will be compiled against an unmodified copy of the provided harness.
3. Do not substitute cuBLAS, CUTLASS, Thrust, or another library implementation for the required GEMM kernels. Standard CUDA runtime APIs are allowed. Tensor Cores, `cp.async`, persistent CTAs, and warp specialization are **not required**.
4. Do not move GEMM, fusion, or output computation to the host. Do not allocate or free GPU memory inside the timed launch wrappers; use the driver-provided scratch/output buffers. Do not alter input tensors. Enqueue all work in the supplied CUDA stream, without per-launch device synchronizations.
5. Each variant must produce the entire packed `Y` for every valid case, including non-divisible dimensions and empty groups. The harness checks numerical correctness; a return value indicating `cudaSuccess` is not sufficient by itself.
6. V1-V3 must use a separate GEMM stage writing `Z` and a separate Bias + ReLU stage writing `Y`. V4 and V5 must not have `Z` for the fused operation. Distinguish performance gains due to fewer launches from gains due to better GEMM computation.

The aim is to discover appropriate mappings and parameters. The tasks below deliberately avoid prescribing grid dimensions, tile shapes, or exact CUDA kernel algorithms.

---

## Variant 1 - Coalesced global-memory access

**Starting point:** the supplied `baseline` (one GEMM stage and one pointwise stage per nonempty group).

**Required improvement:** reconsider how output elements are assigned to threads and how a warp accesses the row-major input/weight/output buffers. Improve spatial locality and memory coalescing while preserving the same stages and all group boundaries.

**Explain in your report:**

- The baseline's memory-access problem and your mapping choice.
- The change in global-memory access efficiency and end-to-end time.
- At least one configuration where the improvement is modest, and your hypothesis for why.

## Variant 2 - Shared-memory reuse

**Starting point:** V1.

**Required improvement:** use GPU shared memory to cooperatively stage data that can be reused by multiple threads. Consider global-memory traffic, synchronization, tile shape, available shared-memory capacity, and bank-conflict behavior. Maintain the separate pointwise stage.

**Explain in your report:**

- Which values are reused and why they justify on-chip storage.
- Evidence from timing and relevant available memory/cache/shared-memory metrics.
- How resource limits and/or boundary tiles influence your design.

## Variant 3 - Register tiling

**Starting point:** V2.

**Required improvement:** exploit thread-private registers to increase data reuse and useful arithmetic per thread. Explore more than one register-work allocation and justify your selected parameters. Maintain the separate Bias + ReLU stage.

**Explain in your report:**

- Your per-thread work granularity and how it changes operand reuse.
- Register usage, occupancy, arithmetic throughput, and any evidence of spills.
- A measured case where more aggressive register tiling does not help (if observed).

## Variant 4 - Fused GEMM + Bias + ReLU

**Starting point:** V3.

**Required improvement:** compute the same mathematical result, but integrate bias and ReLU with GEMM so the intermediate `Z` is not written to and reread from global memory. Keep independent launches for different nonempty groups.

**Explain in your report:**

- What intermediate traffic and launches are avoided compared with V3.
- Why the difference between V3 and V4 may depend on group size.
- The relationship between register-resident results and a fused epilogue.

## Variant 5 - One grouped kernel launch

**Starting point:** V4.

**Required improvement:** preserve the optimized fused computation while executing all nonempty groups via **exactly one CUDA kernel launch** for a problem with at least one nonempty group. Your work assignment must handle different $M_g$, boundary tiles, and empty groups. A valid all-empty problem may enqueue no kernel.

A straightforward rectangular grid may include CTAs with no useful work for some groups. The [scheduling illustration](figures/scheduling.svg) shows the issue; finding a safe and efficient mapping is part of this variant.

**Explain in your report:**

- How the GPU work is divided among groups without per-group launches.
- The effect of group-size imbalance on useful versus wasted scheduling work.
- Evidence from both GPU-event and host wall timings, and an Nsight Systems timeline when available.
- Why V5 may or may not outperform V4 for large or highly skewed groups.

## Optional bonus - Compact or adaptive group scheduling

Improve the V5 grouped schedule to avoid unnecessary work caused by overprovisioning for the largest group. You may examine compact tile assignment or other architecture-portable techniques. A single launch is still required for any nonempty problem. Quantify the benefit **and** the new scheduling/metadata overhead; do not assume fewer CTAs must always be faster.

---

## Workload and measurement requirements

Use the supplied [named cases](docs/INPUT_CONTRACT.md#public-test-cases) and also demonstrate at least one custom case. The report must analyze **small**, **balanced**, **skewed**, and **boundary** workloads and include correctness evidence for **toy**, the illustrated numerical example. The additional `many-tiny`, `medium`, `large`, and `large-skewed` presets are available for optional report experiments.

Correctness is required for every input within the contract, whether or not its case is analyzed in the report. Accordingly, `make check-all` checks more cases than the required report analysis, including `many-tiny`. Running the separate `make check-stress` command is optional; supporting the valid inputs it checks is not. You may profile a smaller representative case if your tool or GPU cannot complete a large profiler run, but disclose that change.

| Comparison | Questions to answer / evidence to seek |
|---|---|
| Baseline -> V1 | Are memory transactions more efficient? Does that translate to lower time? |
| V1 -> V2 | Is there less redundant global traffic? What do shared-memory counters show? |
| V2 -> V3 | Do additional register reuse and instruction parallelism offset occupancy costs? |
| V3 -> V4 | What changes when the intermediate tensor and pointwise launches disappear? |
| V4 -> V5 | How important is launch overhead? What changes with many small or imbalanced groups? |
| V5 -> Bonus | Is reduced scheduling waste worth extra mapping overhead? |

The provided driver reports **median GPU event time**, **median host wall time**, and useful GEMM GFLOP/s. Compare the complete operation, including the pointwise kernel for V1-V3. Do not omit slow cases or report a kernel-only speedup as an end-to-end speedup. If hardware counter collection is unsupported, explain the limitation and rely on valid measurements and reasoned access/occupancy analysis.

### Reproducible performance comparisons

Keep shapes, seeds, compiler flags, warm-up counts, and iteration counts fixed across variants. Verify correctness before collecting timings, and report absolute times alongside speedups. Use ordinary benchmark runs for timing and separate profiler runs for counters and timelines.

### Checking launch structure

Numerical verification does not enforce launch structure. Check the source and, where available, an Nsight Systems CUDA timeline. For a problem with `H` nonempty groups (`H > 0`), the required launch counts are:

| Variant | Required GPU kernel launches per complete operation |
|---|---|
| Baseline and V1-V3 | `2*H`: a GEMM writing `Z` and a separate Bias + ReLU for each nonempty group |
| V4 | `H`: one fused kernel per nonempty group |
| V5 and bonus | Exactly one grouped fused kernel |

Use `--iters=1 --warmup=0` without `--verify` to trace one operation. The `tiny` case has five nonempty groups, giving expected counts of 10, 5, and 1 respectively. The [profiling instructions](README.md#launch-structure-trace) give a trace command.

Also read the code to check that kernels use the provided CUDA stream without synchronizing inside the launcher, each variant follows its required kernel stages, and launchers use the provided buffers without allocating or freeing GPU memory.

### Suggested report outline

1. **Hardware and setup:** GPU model, compute capability, CUDA/tool versions, compiler flags, tested cases.
2. **Variant design:** concise description, chosen parameters, and expected bottleneck addressed.
3. **Correctness:** evidence across regular, boundary, and empty-group cases.
4. **Performance:** a table/plot of baseline and V1-V5 GPU and wall times on multiple workload regimes; speedups measured against both the previous variant and the baseline.
5. **Profiler analysis:** counters/timelines that support or challenge your explanations, not merely raw screenshots.
6. **Trade-offs:** at least one observed regression or limiting scenario and why it occurs (if applicable).
7. **Bonus (if attempted):** scheduling design, added overhead, and measured impact.

Keep the report focused on empirical evidence and GPU architecture. A fast result without a sound explanation is insufficient.

## Evaluation principles

Correctness on the specified input contract is mandatory. Evaluation will consider progressive optimization, quality of profiling and architectural explanation, correctness of the single-launch grouped execution, and code clarity.

Compare performance separately for each workload regime. In particular, evaluate V5 on small and skewed groups rather than relying only on large-GEMM throughput.

## Submission

**Deadline: 22 November 2026, 23:59 IST.**

Submit a folder named with the **last five digits of your SR ID**, containing exactly:

```text
<last_5_digits_of_SR_ID>/
  grouped_gemm_kernels.cu
  report.pdf
```

Do not submit modified copies of `main.cu`, `baseline.cu`, `input_generator.cpp`, `reference.cpp`, headers, binaries, or raw profiler logs. You may include your own design sketches and derived tables inside `report.pdf`.


## Academic Integrity

![Zero Tolerance](https://img.shields.io/badge/ACADEMIC%20DISHONESTY-ZERO%20TOLERANCE-red?style=for-the-badge)

**STRICTLY NO CHEATING IN ANY FORM.**

All submitted code, experimental results, and analysis must be your own work. Plagiarism, copying or sharing solutions, unauthorized collaboration, submitting AI-generated code, and fabricating results are strictly prohibited.

**Any violation will be treated as academic misconduct and handled strictly according to the institute's academic-integrity policy.**
