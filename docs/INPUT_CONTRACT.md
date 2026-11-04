# Input and Evaluation Contract

This document defines the student-facing input space and the exact memory layout. The grader may use additional shapes and seeds **within this contract**. Do not hard-code the public presets or assume every group contains the same number of rows.

## 1. Operation

For group `g = 0, ..., G-1`:

$$
Z_g[i,j] = \sum_{k=0}^{K-1} X_g[i,k] W_g[k,j], \qquad
Y_g[i,j] = \max(0,Z_g[i,j]+b_g[j]).
$$

`0 <= i < M_g`, `0 <= j < N`. `K` and `N` are common to all groups; `M_g` can differ by group. Each `W_g` and `b_g` is independent. All inputs/output are `float` (FP32). Weights are **not transposed**.

V1-V3 write `Z` into a provided scratch buffer and then compute `Y` in a separate kernel. V4/V5 compute the same `Y` without using `Z`.

## 2. Exact packed memory layout

Let `R = sum_g (M_g)` and `row_offsets[0]=0`, `row_offsets[g+1]=row_offsets[g]+M_g`.

| Pointer in `GroupedGemmProblem` | Device/host | Extent | Meaning |
|---|---|---|---|
| `x` | device | `R * K` floats | `X_g[i,k]` at `x[(row_offsets[g] + i) * K + k]` |
| `w` | device | `G * K * N` floats | `W_g[k,j]` at `w[(g * K + k) * N + j]` |
| `bias` | device | `G * N` floats | `b_g[j]` at `bias[g * N + j]` |
| `z` | device | `R * N` floats | GEMM scratch at `z[(row_offsets[g] + i) * N + j]`; `nullptr` for V4/V5/bonus |
| `y` | device | `R * N` floats | Output at `y[(row_offsets[g] + i) * N + j]` |
| `d_row_offsets` | device | `G+1` integers | GPU-readable group boundaries |
| `h_row_offsets` | host | `G+1` integers | CPU-readable identical boundaries |

The fixed-size weight matrices and biases are already packed in group order. `h_row_offsets` is not generally accessible from device code. Input buffers are immutable; `z` may be overwritten by V1-V3, and **every valid element of `y` must be written** during each complete variant invocation, which may contain multiple kernel launches. Different groups' outputs must not overlap.

The driver allocates `z` only for baseline/V1-V3; V4/V5/bonus receive `z == nullptr`. The driver allocates even zero-length logical outputs safely; when `R=0`, no output elements exist and all variants should complete without invalid launches or memory accesses.

### Toy example

With `G=3`, `K=N=2`, `M=[2,1,3]`, the row offsets are `[0,2,3,6]`. The output matrix, stored row-major in a single 12-float buffer, is:

```text
Y = [0,2, 3,5, 2,0, 5,0, 6,0, 0,1]
```

The [overview](../figures/overview.svg) and [packed-memory illustration](../figures/packed-layout.svg) give the exact toy inputs, weights, and biases. The `toy` case ignores the seed and always uses those explicit values.

## 3. Supported shape and memory bounds

| Parameter | Required support |
|---|---|
| `G` | `1..256` |
| `K`, `N` | `1..4096` each |
| Each `M_g` | `0..4096` inclusive |
| Total input rows `R` | `0..65536` |
| Total tensor storage | At most **1 GiB combined**, counting `X`, `W`, `bias`, `Y`, `Z`, and device offsets |
| Layout | Contiguous FP32 row-major, no padding between groups |
| Group composition | Different `M_g` values, including one or more empty groups |

All bounds apply jointly; combinations of individually valid dimensions that exceed the combined storage cap are invalid. The cap counts scratch `Z` for every variant, even when V4/V5/bonus do not allocate it, so all variants share the same supported input domain.

The supported domain includes odd/non-power-of-two values of `K` and `N`, `M_g` smaller than a chosen tile, `M_g` not divisible by a tile, and groups with zero rows. The public presets are **representative**, not the complete test domain.

The kernel launchers must operate asynchronously on the provided CUDA stream. For V5 and bonus, a **nonempty problem requires one GPU kernel launch total**; a problem with `R=0` may require no launch. The driver does not count host metadata computation as a kernel launch, but all launch work contributes to total latency.

## 4. Generated values and reproducibility

For every non-toy case, `X`, `W`, and `bias` contain reproducible pseudo-random, exactly representable FP32 values in `[-1,1]`, including possible zeros and negatives. A seeded xorshift32 stream generates integer samples in `[-64,64]`, which are divided by 64. The default seed is `2026`. A second, independent seeded stream generates custom group sizes, so value generation does not change the group-size distribution.

The public generator implementation in `src/input_generator.cpp` is authoritative. Grading inputs may use different seeds, valid shape lists, and value patterns. **Do not assume fixed data or precomputed output values.**

The checker uses a CPU reference with double-precision accumulation, casts the GEMM sum to FP32, adds the FP32 bias, then applies ReLU. It accepts a finite GPU result `got` if:

```text
abs(got - ref) <= 1e-3 + 1e-3 * abs(ref)
```

All output elements are checked. During `--verify`, output memory is poisoned before the kernel runs to detect unwritten entries. Correctness is not bitwise, and different valid floating-point accumulation orders are permitted within tolerance.

## 5. Public test cases

| `--case` | G | K | N | Group-size pattern / focus |
|---|---:|---:|---:|---|
| `toy` | 3 | 2 | 2 | `[2,1,3]`, exact numerical illustration |
| `tiny` | 7 | 11 | 13 | `[0,1,2,5,7,0,9]`, small/empty/nondivisible |
| `small` | 64 | 256 | 256 | `M_g = 4 + ((13*g) mod 29)`, many small groups |
| `balanced` | 16 | 256 | 256 | `M_g = 64 + 8*(g mod 7)` |
| `skewed` | 32 | 256 | 256 | `M_g = 256+32*(g mod 3)` if `g mod 7=0`, otherwise `0` if `g mod 5=0`, otherwise `4*(1+g mod 4)` |
| `boundary` | 13 | 63 | 70 | `[0,1,2,3,7,15,16,17,31,32,33,63,65]` |
| `wide` | 12 | 512 | 384 | `M_g = 96 + 16*(g mod 5)` |
| `many-tiny` | 256 | 128 | 128 | `M_g = 0` if `g mod 5=0`, otherwise `1 + ((7*g) mod 16)`; launch overhead at the group limit |
| `medium` | 8 | 512 | 512 | `M_g = 256 + 32*(g mod 4)`; balanced compute and reuse |
| `large` | 4 | 2048 | 2048 | Every `M_g = 1024`; larger GEMMs and weight footprint |
| `large-skewed` | 32 | 512 | 512 | `M_g = 4096` if `g mod 8=0`, otherwise `0` if `g mod 5=0`, otherwise `1 + (g mod 16)` |
| `empty` | 4 | 17 | 19 | `[0,0,0,0]`, no work |
| `custom` | variable | variable | variable | Explicit list or seeded patterns, subject to bounds |

Use `./grouped_gemm --list-cases` for the public presets. Named cases determine dimensions and group sizes; `--seed` affects their values except for `toy`. Custom shape flags are accepted only with `--case=custom` to prevent accidentally misinterpreting a preset.

### Routine checks and stress checks

Routine checks cover representative inputs, including cases beyond the required report analysis. The optional stress checks add larger GEMMs and boundary shapes, whose CPU references take longer to compute. These commands check correctness; they do not define which cases must be analyzed in the report. Correctness is required across the full supported domain regardless of which check commands are run.

| Command | Coverage |
|---|---|
| `make test-inputs` | Host generator, validation, and determinism for all presets; full CPU reference checks for routine presets |
| `make check-all` | Baseline and V1-V5 on `toy`, `tiny`, `boundary`, `small`, `balanced`, `skewed`, `wide`, `many-tiny`, and `empty`, plus custom packed-boundary and maximum-row cases; seeds 2026 and 42 |
| `make check-stress` | Routine checks plus `medium`, `large`, `large-skewed`, and custom cases reaching `K=4096` or `N=4096`, with empty groups and partial tiles; stress additions use seed 2026 |

Both GPU check commands verify output and run one timed iteration without warm-up; these timings are diagnostic, not report measurements. Unimplemented variants fail. Use `IMPLS=baseline` to check only the baseline, or `IMPLS=bonus` for the optional bonus.

### Explicit custom group sizes

```bash
./grouped_gemm --case=custom --m-list=0,1,31,2,7 \
  --k=17 --n=19 --seed=42 --impl=baseline --verify
```

The number of entries in `--m-list` determines `G`. All entries must be integers within the supported range.

### Generated custom group sizes

```bash
./grouped_gemm --case=custom --groups=32 \
  --k=256 --n=256 --min-m=0 --max-m=256 \
  --pattern=skewed --seed=42 --impl=baseline --verify
```

Supported custom patterns:

- `uniform`: every group has `M_g = max-m`.
- `random`: each `M_g` is drawn from `[min-m, max-m]`.
- `skewed`: every eighth group has `max-m`; some groups are empty; the rest have small row counts.
- `sparse`: every third group is empty; the remainder use the requested row-count range.

The `--m-list` form takes precedence over `--groups`, `--min-m`, `--max-m`, and `--pattern` (but must still obey all limits). Invalid inputs produce a clear host-side error before kernel launches.

## 6. Timing and evaluation behavior

`--verify` runs an untimed output check. Warm-up iterations run next. For each timed iteration, the driver launches the **entire selected variant** on one stream, records CUDA events around it, and synchronizes at the end. It reports the median GPU-event time and the median host wall time. GEMM throughput is computed from the useful work `2*R*K*N`, excluding ReLU/bias arithmetic. No host-to-device transfer, allocation, data generation, or CPU reference calculation is included in timed iterations.

For V1-V3, the timed unit includes **both** GEMM and pointwise launches for every nonempty group. For V4, it includes all fused per-group launches. For V5, it includes the single grouped fused launch. Comparison on both GPU and wall times is encouraged, especially for many tiny groups.

The student file must not use host-side recomputation, runtime GPU allocations, CUDA graphs, hidden persistent work, or other methods that bypass the specified variants. Output correctness and the required stage/launch structure are both part of the assignment.
