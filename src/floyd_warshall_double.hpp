#pragma once

#include "inf.hpp"

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
double *floyd_warshall_random_init_double(const int n, const double p, const unsigned long seed);

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
// we also need to limit the width and height but keep it a multiple of block_size
double *
floyd_warshall_blocked_random_init_double(const int n, const int block_size, const double p, const unsigned long seed);

// expects len(adjacencyMatrix) == len(distanceMatrix) == n*n
void floyd_warshall_double(double *distanceMatrix, int *successorMatrix, const int n);

// used for blocked_floyd_warshall
#ifdef ISPC
extern "C" void floyd_warshall_in_place_double(double* C, const double* A, const double* B, int *successorMatrix, const int b, const int n);
#else
inline void
floyd_warshall_in_place_double(double *C, const double *A, const double *B, int *successorMatrix, const int b, const int n) {
  for (int k = 0; k < b; k++) {
    int kth = k * n;
    for (int i = 0; i < b; i++) {
      for (int j = 0; j < b; j++) {
        double sum = A[i * n + k] + B[kth + j];
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
floyd_warshall_blocked_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int n, const int b);
extern "C" void free_floyd_warshall_blocked_double(double **distanceMatrix, int **successorMatrix);

#ifdef CUDA
void floyd_warshall_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_blocked_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int **successorMatrix, const int n);
#endif
