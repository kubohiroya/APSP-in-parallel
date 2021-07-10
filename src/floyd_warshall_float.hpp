#pragma once

#include "inf.hpp"

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
float *floyd_warshall_init_float(const int n, const double p, const unsigned long seed);

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
// we also need to limit the width and height but keep it a multiple of block_size
float *floyd_warshall_blocked_init_float(const int n, const int block_size, const double p, const unsigned long seed);

// expects len(input) == len(output) == n*n
void floyd_warshall_float(const float *input, float *output, int *parents, const int n);

// used for blocked_floyd_warshall
#ifdef ISPC
extern "C" void floyd_warshall_in_place_float(float* C, const float* A, const float* B, int *parents, const int b, const int n);
#else

inline void floyd_warshall_in_place_float(float *C, const float *A, const float *B, int *parents, const int b, const int n) {
  for (int k = 0; k < b; k++) {
    int ktn = k * n;
    for (int i = 0; i < b; i++) {
      for (int j = 0; j < b; j++) {
        float sum = A[i * n + k] + B[ktn + j];
        if (C[i * n + j] > sum) {
          C[i * n + j] = sum;
          parents[i * n + j] = parents[ktn + j];
        }
      }
    }
  }
}

#endif

// expects len(input) == len(output) == n*n
extern "C" void floyd_warshall_blocked_float(const float *input, float *output, int *parents, const int n, const int b);

#ifdef CUDA
void floyd_warshall_cuda_float(float* input, float* output, int n);
void floyd_warshall_blocked_cuda_float(float* input, float* output, int n);
#endif

