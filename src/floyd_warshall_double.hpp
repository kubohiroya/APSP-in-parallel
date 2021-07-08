#pragma once

#include "inf.hpp"

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
double *floyd_warshall_init_double(const int n, const double p, const unsigned long seed);

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
// we also need to limit the width and height but keep it a multiple of block_size
double *floyd_warshall_blocked_init_double(const int n, const int block_size, const double p, const unsigned long seed);

// expects len(input) == len(output) == n*n
void floyd_warshall_double(const double *input, double *output, const int n);

// used for blocked_floyd_warshall
#ifdef ISPC
extern "C" void floyd_warshall_in_place_double(double* C, const double* A, const double* B, const int b, const int n);
#else

inline void floyd_warshall_in_place_double(double *C, const double *A, const double *B, const int b, const int n) {
  for (int k = 0; k < b; k++) {
    int kth = k * n;
    for (int i = 0; i < b; i++) {
      for (int j = 0; j < b; j++) {
        double sum = A[i * n + k] + B[kth + j];
        if (C[i * n + j] > sum) {
          C[i * n + j] = sum;
        }
      }
    }
  }
}

#endif

// expects len(input) == len(output) == n*n
void floyd_warshall_blocked_double(const double *input, double *output, const int n, const int b);

#ifdef CUDA
void floyd_warshall_cuda_double(double* input, double* output, int n);
void floyd_warshall_blocked_cuda_double(double* input, double* output, int n);
#endif
