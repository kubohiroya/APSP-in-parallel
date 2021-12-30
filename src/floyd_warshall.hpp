#pragma once

#include <cstring> // memcpy
#include <random> // mt19937_64, uniform_x_distribution

#ifdef _OPENMP
#include "omp.h" // omp_get_num_threads
#endif

#include "inf.hpp"

#ifdef CUDA
template<typename Number> void floyd_warshall_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int **successorMatrix, const int n);
template<typename Number> void floyd_warshall_blocked_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_blocked_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_blocked_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_cuda_int(const int* adjacencyMatrix, int** distanceMatrix, int **successorMatrix, const int n);
void floyd_warshall_blocked_cuda_int(const int* adjacencyMatrix, int** distanceMatrix, int **successorMatrix, const int n);
#endif

#ifdef ISPC
extern "C" void floyd_warshall_in_place_double(double* C, const double* A, const double* B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized);
extern "C" void floyd_warshall_in_place_float(float* C, const float* A, const float* B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized);
extern "C" void floyd_warshall_in_place_int(int* C, const int* A, const int* B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized);

template<typename Number> inline void
floyd_warshall_in_place(Number *C, const Number *A, const Number *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized);
template<> inline void
floyd_warshall_in_place<double>(double *C, const double *A, const double *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_double(C, A, B, successorMatrix, kb, ib, jb, b, n, n_oversized);
}
template<> inline void
floyd_warshall_in_place<float>(float *C, const float *A, const float *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_float(C, A, B, successorMatrix, kb, ib, jb, b, n, n_oversized);
}
template<> inline void
floyd_warshall_in_place<int>(int *C, const int *A, const int *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_int(C, A, B, successorMatrix, kb, ib, jb, b, n, n_oversized);
}

#else
template<typename Number> inline void
floyd_warshall_in_place(Number *C, const Number *A, const Number *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized) {
  for (int k = 0; k < b; k++) {
    for (int i = 0; i < b; i++) {
      int ik = i * n_oversized + k;
      for (int j = 0; j < b; j++) {
        int kj = k * n_oversized + j;
        int ij = i * n_oversized + j;
        double sum = A[ik] + B[kj];
        if (C[ij] > sum) {
          C[ij] = sum;
    	  if(jb + j < n && ib + i < n && kb + k < n){
            successorMatrix[(jb + j) * n + ib + i] = successorMatrix[(jb + j) * n + kb + k];
          }
        }
      }
    }
  }
}
#endif

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
template<typename Number> Number * floyd_warshall_random_init(const int n, const double p, const unsigned long seed, const Number inf);

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
// we also need to limit the width and height but keep it a multiple of block_size
template<typename Number> Number * floyd_warshall_blocked_random_init(const int n, const int block_size, const double p, const unsigned long seed, const Number inf);

// expects len(adjacencyMatrix) == len(distanceMatrix) == n*n
template<typename Number> void floyd_warshall(Number *distanceMatrix, int *successorMatrix, const int n, const Number inf);

template<typename Number> Number * floyd_warshall_random_init(const int n, const double p, const unsigned long seed, Number inf) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_real_distribution<double> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  Number *out = new Number[n * n];
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        out[i * n + j] = 0.0;
      } else if (flip(rand_engine) < p) {
        out[i * n + j] = choose_weight(rand_engine) * 1.0;
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n + j] = inf;
      }
    }
  }

  return out;
}

template<typename Number> Number *
floyd_warshall_blocked_random_init(const int n, const int block_size, const double p, const unsigned long seed, const Number inf) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_real_distribution<double> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int block_remainder = n % block_size;
  int n_oversized = (block_remainder == 0) ? n : n + block_size - block_remainder;

  Number *out = new Number[n_oversized * n_oversized];
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n_oversized; i++) {
    for (int j = 0; j < n_oversized; j++) {
      if (i == j) {
        out[i * n_oversized + j] = 0.0;
      } else if (i < n && j < n && flip(rand_engine) < p) {
        out[i * n_oversized + j] = choose_weight(rand_engine);
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n_oversized + j] = inf;
      }
    }
  }

  return out;
}

template<typename Number> void floyd_warshall(Number *distanceMatrix, int *successorMatrix, const int n) {
  for (int k = 0; k < n; k++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        Number sum = distanceMatrix[i * n + k] + distanceMatrix[k * n + j];
        if (distanceMatrix[i * n + j] > sum) {
          distanceMatrix[i * n + j] = sum;
          successorMatrix[j * n + i] = successorMatrix[j * n + k];
        }
      }
    }
  }
}

template<typename Number> void _floyd_warshall_blocked(Number *distanceMatrix, int *successorMatrix, const int b, const int n, const int n_oversized) {
  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * adjacencyMatrix_width * block_width + j * block_width]
  for (int k = 0; k < n_oversized; k += b) {
    int kk = k * n_oversized + k;
    floyd_warshall_in_place<Number>(&distanceMatrix[kk], &distanceMatrix[kk], &distanceMatrix[kk],
                                successorMatrix, k, k, k, b, n, n_oversized);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < n_oversized; j += b) {
      if (j == k) continue;
      int kj = k * n_oversized + j;
      floyd_warshall_in_place<Number>(&distanceMatrix[kj], &distanceMatrix[kk], &distanceMatrix[kj],
                                  successorMatrix, k, k, j, b, n, n_oversized);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n_oversized; i += b) {
      if (i == k) continue;
      int ik = i * n_oversized + k;
      floyd_warshall_in_place<Number>(&distanceMatrix[ik], &distanceMatrix[ik], &distanceMatrix[kk],
                                  successorMatrix, k, i, k, b, n, n_oversized);
      for (int j = 0; j < n_oversized; j += b) {
        if (j == k) continue;
        int ij = i * n_oversized + j;
        int kj = k * n_oversized + j;
        floyd_warshall_in_place<Number>(&distanceMatrix[ij], &distanceMatrix[ik], &distanceMatrix[kj],
                                    successorMatrix, k, i, j, b, n, n_oversized);
      }
    }
  }
}

template<typename Number> void floyd_warshall_blocked(const Number *adjacencyMatrix, Number **distanceMatrix, int **successorMatrix, const int b, const int n, const Number inf) {
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);
  std::memcpy(*distanceMatrix, adjacencyMatrix, sizeof(Number) * n * n);
  *successorMatrix = (int *) malloc(sizeof(int) * n * n);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      (*successorMatrix)[i * n + j] = j;
    }
  }

#ifdef CUDA
  floyd_warshall_blocked_cuda<Number>((Number *)adjacencyMatrix, distanceMatrix, successorMatrix, n);
  return;
#else
  if(b == -1 || n <= b) {
    floyd_warshall<Number>(*distanceMatrix, *successorMatrix, n);
    return;
  }

  int block_remainder = n % b;
  if(block_remainder == 0){
    _floyd_warshall_blocked<Number>(*distanceMatrix, *successorMatrix, b, n, n);
    return;
  }

  int n_oversized = n + b - block_remainder;
  Number *_distanceMatrix = new Number[n_oversized * n_oversized];

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      _distanceMatrix[i * n_oversized + j] = (*distanceMatrix)[i * n + j];
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = n; i < n_oversized; i++) {
    for (int j = 0; j < n; j++) {
      _distanceMatrix[i * n_oversized + j] = inf;
      _distanceMatrix[j * n_oversized + i] = inf;
    }
 }
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = n; i < n_oversized; i++) {
    for (int j = n; j < n_oversized; j++) {
      _distanceMatrix[i * n_oversized + j] = inf;
    }
    _distanceMatrix[i * n_oversized + i] = 0;
 }

 _floyd_warshall_blocked<Number>(_distanceMatrix, *successorMatrix, b, n, n_oversized);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      (*distanceMatrix)[i * n + j] = _distanceMatrix[i * n_oversized + j];
    }
  }
  delete[] _distanceMatrix;
#endif
}

template<typename Number> void free_floyd_warshall_blocked(Number **distanceMatrix, int **successorMatrix) {
  free(*distanceMatrix);
  free(*successorMatrix);
}

// expects len(adjacencyMatrix) == len(distanceMatrix) == n*n
extern "C" void floyd_warshall_blocked_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_double(double **distanceMatrix, int **successorMatrix);
extern "C" void floyd_warshall_blocked_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_float(float **distanceMatrix, int **successorMatrix);
extern "C" void floyd_warshall_blocked_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_int(int **distanceMatrix, int **successorMatrix);
