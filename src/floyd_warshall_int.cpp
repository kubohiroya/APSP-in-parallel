#include <cstring> // memcpy
#include <iostream> // cout
#include <random> // mt19937_64, uniform_x_distribution

#ifdef _OPENMP
#include "omp.h" // omp_get_num_threads
#endif

#include "floyd_warshall_int.hpp"

int *floyd_warshall_random_init_int(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int *out = new int[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        out[i * n + j] = 0;
      } else if (flip(rand_engine) < p) {
        out[i * n + j] = choose_weight(rand_engine);
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n + j] = INT_INF;
      }
    }
  }

  return out;
}

int *
floyd_warshall_blocked_random_init_int(const int n, const int block_size, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int block_remainder = n % block_size;
  int n_oversized = (block_remainder == 0) ? n : n + block_size - block_remainder;

  int *out = new int[n_oversized * n_oversized];
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n_oversized; i++) {
    for (int j = 0; j < n_oversized; j++) {
      if (i == j) {
        out[i * n_oversized + j] = 0;
      } else if (i < n && j < n && flip(rand_engine) < p) {
        out[i * n_oversized + j] = choose_weight(rand_engine);
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n_oversized + j] = INT_INF;
      }
    }
  }

  return out;
}

void floyd_warshall_int(int *distanceMatrix, int *successorMatrix, const int n) {
  for (int k = 0; k < n; k++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        int sum = distanceMatrix[i * n + k] + distanceMatrix[k * n + j];
        if (distanceMatrix[i * n + j] > sum) {
          distanceMatrix[i * n + j] = sum;
          successorMatrix[j * n + i] = successorMatrix[j * n + k];
        }
      }
    }
  }
}

void _floyd_warshall_blocked_int(int *distanceMatrix, int *successorMatrix, const int n, const int b) {
  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * adjacencyMatrix_width * block_width + j * block_width]
  for (int k = 0; k < blocks; k++) {
    int kb = k * b;
    int kk = kb * n + kb;
    floyd_warshall_in_place_int(&distanceMatrix[kk], &distanceMatrix[kk], &distanceMatrix[kk],
                                successorMatrix, kb, kb, kb, b, n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < blocks; j++) {
      if (j == k) continue;
      int jb = j * b;
      int kj = kb * n + jb;
      floyd_warshall_in_place_int(&distanceMatrix[kj], &distanceMatrix[kk], &distanceMatrix[kj],
                                  successorMatrix, kb, kb, jb, b, n);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < blocks; i++) {
      if (i == k) continue;
      int ib = i * b;
      int ik = ib * n + kb;
      floyd_warshall_in_place_int(&distanceMatrix[ik], &distanceMatrix[ik], &distanceMatrix[kk],
                                  successorMatrix, kb, ib, kb, b, n);
      for (int j = 0; j < blocks; j++) {
        if (j == k) continue;
        int jb = j * b;
        int ij = ib * n + jb;
        int kj = kb * n + jb;
        floyd_warshall_in_place_int(&distanceMatrix[ij], &distanceMatrix[ik], &distanceMatrix[kj],
                                    successorMatrix, kb, ib, jb, b, n);
      }
    }
  }
}

void floyd_warshall_blocked_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int n, const int b) {
  *distanceMatrix = (int *) malloc(sizeof(int) * n * n);
  std::memcpy(*distanceMatrix, adjacencyMatrix, sizeof(int) * n * n);
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
  floyd_warshall_blocked_cuda_int(adjacencyMatrix, distanceMatrix, successorMatrix, n);
#else
  if(b != -1 && n > b) {
      int block_remainder = n % b;
      int n_oversized = (block_remainder == 0) ? n : n + b - block_remainder;
      int *_distanceMatrix = new int[n_oversized * n_oversized];

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
        _distanceMatrix[i * n_oversized + j] = INT_INF;
        _distanceMatrix[j * n_oversized + i] = INT_INF;
      }
   }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = n; i < n_oversized; i++) {
      for (int j = n; j < n_oversized; j++) {
        _distanceMatrix[i * n_oversized + j] = INT_INF;
      }
   }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = n; i < n_oversized; i++) {
      _distanceMatrix[i * n_oversized + i] = 0;
    }

    _floyd_warshall_blocked_int(_distanceMatrix, *successorMatrix, n, b);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        (*distanceMatrix)[i * n + j] = _distanceMatrix[i * n_oversized + j];
      }
    }

    delete[] _distanceMatrix;

  }else{
      floyd_warshall_int(*distanceMatrix, *successorMatrix, n);
  }
#endif
}

void free_floyd_warshall_blocked_int(int **distanceMatrix, int **successorMatrix) {
  free(*distanceMatrix);
  free(*successorMatrix);
}
