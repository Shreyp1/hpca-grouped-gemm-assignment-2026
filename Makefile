NVCC ?= nvcc
CXX ?= g++
ARCH ?=
DEBUG ?= 0

ARCHFLAGS :=
ifneq ($(strip $(ARCH)),)
ARCHFLAGS := -gencode arch=compute_$(ARCH),code=sm_$(ARCH)
endif
NVCCFLAGS := -std=c++17 -lineinfo -O3 -Iinclude $(ARCHFLAGS)
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude
ifeq ($(DEBUG),1)
NVCCFLAGS := -std=c++17 -O0 -g -G -Iinclude $(ARCHFLAGS)
endif

SRCS := src/main.cu src/baseline.cu src/grouped_gemm_kernels.cu \
        src/input_generator.cpp src/reference.cpp
HEADERS := include/grouped_gemm.h include/input_generator.h

.PHONY: all clean run check check-all check-stress test-inputs help
all: grouped_gemm

grouped_gemm: $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@

tests/test_inputs: tests/test_inputs.cpp src/input_generator.cpp src/reference.cpp include/input_generator.h
	$(CXX) $(CXXFLAGS) tests/test_inputs.cpp src/input_generator.cpp src/reference.cpp -o $@

test-inputs: tests/test_inputs
	./tests/test_inputs

run: grouped_gemm
	./grouped_gemm --case=toy --impl=baseline --verify

check: grouped_gemm test-inputs
	./grouped_gemm --case=toy --impl=baseline --verify --iters=2
	./grouped_gemm --case=boundary --impl=baseline --verify --iters=2
	./grouped_gemm --case=empty --impl=baseline --verify --iters=2

check-all: grouped_gemm
	./scripts/check_all.sh

check-stress: grouped_gemm
	./scripts/check_all.sh stress

help:
	@echo 'make                  Build benchmark (requires nvcc)'
	@echo 'make ARCH=75          Build for RTX 2080 Ti (compute capability 7.5)'
	@echo 'make DEBUG=1          Debug build'
	@echo 'make test-inputs      Host-only input/reference tests (no CUDA needed)'
	@echo 'make check            CUDA baseline smoke tests'
	@echo 'make check-all        Routine GPU checks, two seeds (TODOs fail)'
	@echo 'make check-stress     Routine plus expensive large/boundary checks'
	@echo 'make clean            Remove built artifacts'

clean:
	rm -f grouped_gemm tests/test_inputs
