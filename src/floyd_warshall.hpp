#pragma once

#include <cstring> // memcpy
#include <iostream> // cout

#ifdef _OPENMP
#include "omp.h" // omp_get_num_threads
#endif

#include "util.hpp"
#include "getInf.hpp"

#ifdef CUDA
template<typename Number> void floyd_warshall_blocked_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int n);
void floyd_warshall_blocked_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int n);
void floyd_warshall_blocked_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int n);
void floyd_warshall_blocked_cuda_int(const int* adjacencyMatrix, int** distanceMatrix, int n);
template<typename Number> void floyd_warshall_blocked_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int **successorMatrix, int n);
void floyd_warshall_blocked_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int **successorMatrix, int n);
void floyd_warshall_blocked_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int **successorMatrix, int n);
void floyd_warshall_blocked_cuda_int(const int* adjacencyMatrix, int** distanceMatrix, int **successorMatrix, int n);
template<typename Number> void floyd_warshall_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int n);
void floyd_warshall_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int n);
void floyd_warshall_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int n);
void floyd_warshall_cuda_int(const int* adjacencyMatrix, int** distanceMatrix, int n);
template<typename Number> void floyd_warshall_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int **successorMatrix, int n);
void floyd_warshall_cuda_double(const double* adjacencyMatrix, double** distanceMatrix, int **successorMatrix, int n);
void floyd_warshall_cuda_float(const float* adjacencyMatrix, float** distanceMatrix, int **successorMatrix, int n);
void floyd_warshall_cuda_int(const int* adjacencyMatrix, int** distanceMatrix, int **successorMatrix, int n);
#endif

#ifdef ISPC
extern "C" void floyd_warshall_in_place_double(double* C, const double* A, const double* B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized, const double inf);
extern "C" void floyd_warshall_in_place_float(float* C, const float* A, const float* B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized, const float inf);
extern "C" void floyd_warshall_in_place_int(int* C, const int* A, const int* B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized, const int inf);
extern "C" void floyd_warshall_in_place_successor_double(double* C, const double* A, const double* B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized, const double inf);
extern "C" void floyd_warshall_in_place_successor_float(float* C, const float* A, const float* B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized, const float inf);
extern "C" void floyd_warshall_in_place_successor_int(int* C, const int* A, const int* B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized, const int inf);

template<typename Number> inline void
floyd_warshall_in_place(Number *C, const Number *A, const Number *B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized);
template<> inline void
floyd_warshall_in_place<double>(double *C, const double *A, const double *B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_double(C, A, B, kb, ib, jb, b, n, n_oversized, DBL_INF);
}
template<> inline void
floyd_warshall_in_place<float>(float *C, const float *A, const float *B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_float(C, A, B, kb, ib, jb, b, n, n_oversized, FLT_INF);
}
template<> inline void
floyd_warshall_in_place<int>(int *C, const int *A, const int *B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_int(C, A, B, kb, ib, jb, b, n, n_oversized, INT_INF);
}

template<typename Number> inline void
floyd_warshall_in_place(Number *C, const Number *A, const Number *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized);
template<> inline void
floyd_warshall_in_place<double>(double *C, const double *A, const double *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_successor_double(C, A, B, successorMatrix, kb, ib, jb, b, n, n_oversized, DBL_INF);
}
template<> inline void
floyd_warshall_in_place<float>(float *C, const float *A, const float *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_successor_float(C, A, B, successorMatrix, kb, ib, jb, b, n, n_oversized, FLT_INF);
}
template<> inline void
floyd_warshall_in_place<int>(int *C, const int *A, const int *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized){
  floyd_warshall_in_place_successor_int(C, A, B, successorMatrix, kb, ib, jb, b, n, n_oversized, INT_INF);
}

#else
template<typename Number> inline void
floyd_warshall_in_place(Number *C, const Number *A, const Number *B, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized) {
  static const Number inf = getInf<Number>();
  for (int k = 0; k < b; k++) {
    for (int i = 0; i < b; i++) {
      int ik = i * n_oversized + k;
      for (int j = 0; j < b; j++) {
        int kj = k * n_oversized + j;
        int ij = i * n_oversized + j;
        if (A[ik] != inf && B[kj] != inf) {
          double sum = A[ik] + B[kj];
          if (C[ij] > sum) {
            C[ij] = sum;
          }
        }
      }
    }
  }
}

template<typename Number> inline void
floyd_warshall_in_place(Number *C, const Number *A, const Number *B, int *successorMatrix, const int kb, const int ib, const int jb, const int b, const int n, const int n_oversized) {
  static const Number inf = getInf<Number>();
  for (int k = 0; k < b; k++) {
    for (int i = 0; i < b; i++) {
      int ik = i * n_oversized + k;
      for (int j = 0; j < b; j++) {
        int kj = k * n_oversized + j;
        int ij = i * n_oversized + j;
        if (A[ik] != inf && B[kj] != inf) {
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
}
#endif

template<typename Number> void floyd_warshall(const Number *adjacencyMatrix, Number **distanceMatrix, const int n) {
  static const Number inf = getInf<Number>();
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);
  std::memcpy(*distanceMatrix, adjacencyMatrix, sizeof(Number) * n * n);

  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if((*distanceMatrix)[i * n + k] != inf && (*distanceMatrix)[k * n + j] != inf){
          Number sum = (*distanceMatrix)[i * n + k] + (*distanceMatrix)[k * n + j];
          if ((*distanceMatrix)[i * n + j] > sum) {
            (*distanceMatrix)[i * n + j] = sum;
          }
        }
      }
    }
  }
}

template<typename Number> void floyd_warshall(const Number *adjacencyMatrix, Number **distanceMatrix, int **successorMatrix, const int n) {
  static const Number inf = getInf<Number>();
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);
  std::memcpy(*distanceMatrix, adjacencyMatrix, sizeof(Number) * n * n);
  *successorMatrix = (int *) malloc(sizeof(int) * n * n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      (*successorMatrix)[i * n + j] = j;
    }
  }

  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if((*distanceMatrix)[i * n + k] != inf && (*distanceMatrix)[k * n + j] != inf){
          Number sum = (*distanceMatrix)[i * n + k] + (*distanceMatrix)[k * n + j];
          if ((*distanceMatrix)[i * n + j] > sum) {
            (*distanceMatrix)[i * n + j] = sum;
            (*successorMatrix)[j * n + i] = (*successorMatrix)[j * n + k];
          }
        }
      }
    }
  }
}

template<typename Number> void _floyd_warshall_blocked(Number *distanceMatrix, const int b, const int n, const int n_oversized) {
  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * adjacencyMatrix_width * block_width + j * block_width]
  for (int k = 0; k < n_oversized; k += b) {
    int kk = k * n_oversized + k;
    floyd_warshall_in_place<Number>(&distanceMatrix[kk], &distanceMatrix[kk], &distanceMatrix[kk],
                                k, k, k, b, n, n_oversized);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < n_oversized; j += b) {
      if (j == k) continue;
      int kj = k * n_oversized + j;
      floyd_warshall_in_place<Number>(&distanceMatrix[kj], &distanceMatrix[kk], &distanceMatrix[kj],
                                  k, k, j, b, n, n_oversized);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n_oversized; i += b) {
      if (i == k) continue;
      int ik = i * n_oversized + k;
      floyd_warshall_in_place<Number>(&distanceMatrix[ik], &distanceMatrix[ik], &distanceMatrix[kk],
                                  k, i, k, b, n, n_oversized);
      for (int j = 0; j < n_oversized; j += b) {
        if (j == k) continue;
        int ij = i * n_oversized + j;
        int kj = k * n_oversized + j;
        floyd_warshall_in_place<Number>(&distanceMatrix[ij], &distanceMatrix[ik], &distanceMatrix[kj],
                                  k, i, j, b, n, n_oversized);
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

/*
template<typename Number> void floyd_warshall_blocked(const Number *adjacencyMatrix, Number **distanceMatrix, const int n, const int b) {
  floyd_warshall_blocked<Number>(adjacencyMatrix, (Number**)nullptr, n, b);
}
*/

template<typename Number> void floyd_warshall_blocked(const Number *adjacencyMatrix, Number **distanceMatrix, const int n, const int b) {
  static const Number inf = getInf<Number>();
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);

#ifdef CUDA
  floyd_warshall_cuda<Number>(adjacencyMatrix, distanceMatrix, n);
  // floyd_warshall_blocked_cuda<Number>(adjacencyMatrix, distanceMatrix, n);
  return;
#else
  if(b == -1 || n <= b) {
    floyd_warshall<Number>(adjacencyMatrix, distanceMatrix, n);
    return;
  }

  int block_remainder = n % b;
  if(block_remainder == 0){
      std::memcpy(*distanceMatrix, adjacencyMatrix, sizeof(Number) * n * n);
      _floyd_warshall_blocked<Number>(*distanceMatrix, b, n, n);
      return;
    }

  int n_oversized = n + b - block_remainder;
  Number *_distanceMatrix = new Number[n_oversized * n_oversized];

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      _distanceMatrix[i * n_oversized + j] = adjacencyMatrix[i * n + j];
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

 _floyd_warshall_blocked<Number>(_distanceMatrix, b, n, n_oversized);

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

template<typename Number> void floyd_warshall_blocked(const Number *adjacencyMatrix, Number **distanceMatrix, int **successorMatrix, const int n, const int b) {
  static const Number inf = getInf<Number>();
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);
  if(successorMatrix != nullptr)
    *successorMatrix = (int *) malloc(sizeof(int) * n * n);

  if(successorMatrix != nullptr){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        (*successorMatrix)[i * n + j] = j;
      }
    }
  }

#ifdef CUDA
  floyd_warshall_cuda<Number>((Number *)adjacencyMatrix, distanceMatrix, successorMatrix, n);
  // floyd_warshall_blocked_cuda<Number>((const Number*)adjacencyMatrix, distanceMatrix, successorMatrix, n);
  return;
#else
  if(b == -1 || n <= b) {
    floyd_warshall<Number>(adjacencyMatrix, distanceMatrix, successorMatrix, n);
    return;
  }

  int block_remainder = n % b;
  if(block_remainder == 0){
    std::memcpy(*distanceMatrix, adjacencyMatrix, sizeof(Number) * n * n);
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
      _distanceMatrix[i * n_oversized + j] = adjacencyMatrix[i * n + j];
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

 _floyd_warshall_blocked<Number>(_distanceMatrix, (int *)*successorMatrix, b, n, n_oversized);

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

template<typename Number> void free_floyd_warshall_blocked(Number **distanceMatrix) {
  free(*distanceMatrix);
}

template<typename Number> void free_floyd_warshall_blocked(Number **distanceMatrix, int **successorMatrix) {
  free(*distanceMatrix);
  free(*successorMatrix);
}

extern "C" void floyd_warshall_blocked_double(const double *adjacencyMatrix, double **distanceMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_double(double **distanceMatrix);
extern "C" void floyd_warshall_blocked_float(const float *adjacencyMatrix, float **distanceMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_float(float **distanceMatrix);
extern "C" void floyd_warshall_blocked_int(const int *adjacencyMatrix, int **distanceMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_int(int **distanceMatrix);

extern "C" void floyd_warshall_blocked_successor_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_successor_double(double **distanceMatrix, int **successorMatrix);
extern "C" void floyd_warshall_blocked_successor_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_successor_float(float **distanceMatrix, int **successorMatrix);
extern "C" void floyd_warshall_blocked_successor_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int b, const int n);
extern "C" void free_floyd_warshall_blocked_successor_int(int **distanceMatrix, int **successorMatrix);
