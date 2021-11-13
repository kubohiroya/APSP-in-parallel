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

void floyd_warshall_double(double *output, int *parents, const int n) {
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (output[i * n + j] > output[i * n + k] + output[k * n + j]) {
          output[i * n + j] = output[i * n + k] + output[k * n + j];
          parents[i * n + j] = parents[i * n + k];
        }
      }
    }
  }
}

void _floyd_warshall_blocked_double(double *output, int *parents, const int n, const int b) {
  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * input_width * block_width + j * block_width]
  for (int k = 0; k < blocks; k++) {
    int kbnkb = k * b * n + k * b;
    floyd_warshall_in_place_double(&output[kbnkb], &output[kbnkb], &output[kbnkb],
                                   &parents[kbnkb], b, n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < blocks; j++) {
      if (j == k) continue;
      int kbnjb = k * b * n + j * b;
      floyd_warshall_in_place_double(&output[kbnjb], &output[kbnkb], &output[kbnjb], &parents[kbnjb], b, n);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < blocks; i++) {
      if (i == k) continue;
      int ibnkb = i * b * n + k * b;
      floyd_warshall_in_place_double(&output[ibnkb], &output[ibnkb], &output[ibnkb], &parents[ibnkb], b, n);
      for (int j = 0; j < blocks; j++) {
        if (j == k) continue;
        int ibnjb = i * b * n + j * b;
        floyd_warshall_in_place_double(&output[ibnjb], &output[ibnkb],
                                       &output[k * b * n + j * b], &parents[ibnjb], b, n);
      }
    }
  }
}

void floyd_warshall_blocked_double(const double *input, double **output, int **parents, const int n, const int b) {
  *output = (double *) malloc(sizeof(double) * n * n);
  std::memcpy(*output, input, sizeof(double) * n * n);
  *parents = (int *) malloc(sizeof(int) * n * n);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        (*parents)[i * n + j] = j;
    }
  }
  if(n >= b) {
    _floyd_warshall_blocked_double(*output, *parents, n, b);
  }else{
    floyd_warshall_double(*output, *parents, n);
  }
}

void free_floyd_warshall_blocked_double(double *output, int *parents) {
  free(output);
  free(parents);
}
