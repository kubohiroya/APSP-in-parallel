#include <cstring> // memcpy
#include <random> // mt19937_64, uniform_x_distribution

#ifdef _OPENMP
#include "omp.h" // omp_get_num_threads
#endif

#include "floyd_warshall_double.hpp"

double *floyd_warshall_random_init_double(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  double *out = new double[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        out[i * n + j] = 0.0;
      } else if (flip(rand_engine) < p) {
        out[i * n + j] = choose_weight(rand_engine) * 1.0;
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n + j] = DBL_INF;
      }
    }
  }

  return out;
}

double *
floyd_warshall_blocked_random_init_double(const int n, const int block_size, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int block_remainder = n % block_size;
  int n_oversized = (block_remainder == 0) ? n : n + block_size - block_remainder;

  double *out = new double[n_oversized * n_oversized];
  for (int i = 0; i < n_oversized; i++) {
    for (int j = 0; j < n_oversized; j++) {
      if (i == j) {
        out[i * n_oversized + j] = 0.0;
      } else if (i < n && j < n && flip(rand_engine) < p) {
        out[i * n_oversized + j] = choose_weight(rand_engine) * 1.0;
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n_oversized + j] = DBL_INF;
      }
    }
  }

  return out;
}

void floyd_warshall_double(double *distanceMatrix, int *successorMatrix, const int n) {
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (distanceMatrix[i * n + j] > distanceMatrix[i * n + k] + distanceMatrix[k * n + j]) {
          distanceMatrix[i * n + j] = distanceMatrix[i * n + k] + distanceMatrix[k * n + j];
          successorMatrix[i * n + j] = successorMatrix[i * n + k];
        }
      }
    }
  }
}

void _floyd_warshall_blocked_double(double *distanceMatrix, int *successorMatrix, const int n, const int b) {
  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * adjacency_width * block_width + j * block_width]
  for (int k = 0; k < blocks; k++) {
    int kbnkb = k * b * n + k * b;
    floyd_warshall_in_place_double(&distanceMatrix[kbnkb], &distanceMatrix[kbnkb], &distanceMatrix[kbnkb],
                                   &successorMatrix[kbnkb], b, n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < blocks; j++) {
      if (j == k) continue;
      int kbnjb = k * b * n + j * b;
      floyd_warshall_in_place_double(&distanceMatrix[kbnjb], &distanceMatrix[kbnkb], &distanceMatrix[kbnjb], &successorMatrix[kbnjb], b, n);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < blocks; i++) {
      if (i == k) continue;
      int ibnkb = i * b * n + k * b;
      floyd_warshall_in_place_double(&distanceMatrix[ibnkb], &distanceMatrix[ibnkb], &distanceMatrix[ibnkb], &successorMatrix[ibnkb], b, n);
      for (int j = 0; j < blocks; j++) {
        if (j == k) continue;
        int ibnjb = i * b * n + j * b;
        floyd_warshall_in_place_double(&distanceMatrix[ibnjb], &distanceMatrix[ibnkb],
                                       &distanceMatrix[k * b * n + j * b], &successorMatrix[ibnjb], b, n);
      }
    }
  }
}

void floyd_warshall_blocked_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int n, const int b) {
  *distanceMatrix = (double *) malloc(sizeof(double) * n * n);
  std::memcpy(*distanceMatrix, adjacencyMatrix, sizeof(double) * n * n);
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
  floyd_warshall_blocked_cuda_double(adjacencyMatrix, *distanceMatrix, *successorMatrix, n);
#else
  if(n >= b) {
    _floyd_warshall_blocked_double(*distanceMatrix, *successorMatrix, n, b);
  }else{
    floyd_warshall_double(*distanceMatrix, *successorMatrix, n);
  }
#endif
}

void free_floyd_warshall_blocked_double(double **distanceMatrix, int **successorMatrix) {
  free(*distanceMatrix);
  free(*successorMatrix);
}
