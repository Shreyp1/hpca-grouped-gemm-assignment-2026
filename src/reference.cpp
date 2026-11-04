#include "input_generator.h"

#include <algorithm>
#include <vector>

std::vector<float> cpu_reference(const HostProblem& input) {
    const int total_rows = input.total_rows();
    std::vector<float> output(static_cast<size_t>(total_rows) * input.n, 0.0f);
    for (int g = 0; g < input.groups; ++g) {
        const int begin = input.row_offsets[g];
        const int end = input.row_offsets[g + 1];
        for (int global_row = begin; global_row < end; ++global_row) {
            for (int col = 0; col < input.n; ++col) {
                double sum = 0.0;
                for (int kk = 0; kk < input.k; ++kk) {
                    sum += static_cast<double>(input.x[static_cast<size_t>(global_row) * input.k + kk]) *
                           static_cast<double>(input.w[(static_cast<size_t>(g) * input.k + kk) * input.n + col]);
                }
                const float gemm = static_cast<float>(sum);
                const float biased = gemm + input.bias[static_cast<size_t>(g) * input.n + col];
                output[static_cast<size_t>(global_row) * input.n + col] = std::max(0.0f, biased);
            }
        }
    }
    return output;
}
