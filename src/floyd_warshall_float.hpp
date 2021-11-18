#pragma once

#include "inf.hpp"

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
float *floyd_warshall_random_init_float(const int n, const double p, const unsigned long seed);

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
// we also need to limit the width and height but keep it a multiple of block_size
float *
floyd_warshall_blocked_random_init_float(const int n, const int block_size, const double p, const unsigned long seed);

// expects len(adjacencyMatrix) == len(distanceMatrix) == n*n
void floyd_warshall_float(float *distanceMatrix, int *successorMatrix, const int n);

// used for blocked_floyd_warshall
#ifdef ISPC
extern "C" void floyd_warshall_in_place_float(float* C, const float* A, const float* B, int *successorMatrix, const int b, const int n);
#else

inline void
floyd_warshall_in_place_float(float *C, const float *A, const float *B, int *successorMatrix, const int b, const int n) {
  for (int k = 0; k < b; k++) {
    int ktn = k * n;
    for (int i = 0; i < b; i++) {
      for (int j = 0; j < b; j++) {
        float sum = A[i * n + k] + B[ktn + j];
        if (C[i * n + j] > sum) {
          C[i * n + j] = sum;
          successorMatrix[i * n + j] = successorMatrix[i * n + k];
        }
      }
    }
  }
}

#endif

// expects len(adjacencyMatrix) == len(distanceMatrix) == n*n
extern "C" void
floyd_warshall_blocked_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int n, const int b);
extern "C" void free_floyd_warshall_blocked_float(float **distanceMatrix, int **successorMatrix);

#ifdef CUDA
void floyd_warshall_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_blocked_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int **successorMatrix, const int n);
#endif

