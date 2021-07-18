#include <cstring> // memcpy
#include <random> // mt19937_64, uniform_x_distribution

#ifdef _OPENMP
#include "omp.h" // omp_get_num_threads
#endif

#include "floyd_warshall_float.hpp"

float *floyd_warshall_init_float(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  float *out = new float[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        out[i * n + j] = 0.0f;
      } else if (flip(rand_engine) < p) {
        out[i * n + j] = choose_weight(rand_engine) * 1.0f;
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n + j] = FLT_INF;
      }
    }
  }

  return out;
}

float *floyd_warshall_blocked_init_float(const int n, const int block_size, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int n_oversized;
  int block_remainder = n % block_size;
  if (block_remainder == 0) {
    n_oversized = n;
  } else {
    n_oversized = n + block_size - block_remainder;
  }

  float *out = new float[n_oversized * n_oversized];
  for (int i = 0; i < n_oversized; i++) {
    for (int j = 0; j < n_oversized; j++) {
      if (i == j) {
        out[i * n_oversized + j] = 0.0f;
      } else if (i < n && j < n && flip(rand_engine) < p) {
        out[i * n_oversized + j] = choose_weight(rand_engine);
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i * n_oversized + j] = FLT_INF;
      }
    }
  }

  return out;
}

void floyd_warshall_float(const float *input, float *output, int *parents, const int n) {
  std::memcpy(output, input, n * n * sizeof(float));
  std::memset(parents, -1, n * n * sizeof(int));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      parents[i * n + j] = i;
    }
  }
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (output[i * n + j] > output[i * n + k] + output[k * n + j]) {
          output[i * n + j] = output[i * n + k] + output[k * n + j];
          parents[i * n + j] = parents[k * n + j];
        }
      }
    }
  }
}

void floyd_warshall_blocked_float(const float *input, float *output, int *parents, const int n, const int b) {
  std::memcpy(output, input, n * n * sizeof(float));
  std::memset(parents, -1, n * n * sizeof(int));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      parents[i * n + j] = i;
    }
  }
  // for now, assume b divides n
  const int blocks = n / b;

  // note that [i][j] == [i * input_width * block_width + j * block_width]
  for (int k = 0; k < blocks; k++) {
    int kbnkb = k * b * n + k * b;
    floyd_warshall_in_place_float(&output[kbnkb], &output[kbnkb], &output[kbnkb], &parents[kbnkb], b, n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < blocks; j++) {
      if (j == k) continue;
      int kbnjb = k * b * n + j * b;
      floyd_warshall_in_place_float(&output[kbnjb], &output[kbnkb], &output[kbnjb], &parents[kbnjb], b, n);
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < blocks; i++) {
      if (i == k) continue;
      int ibnkb = i * b * n + k * b;
      floyd_warshall_in_place_float(&output[ibnkb], &output[ibnkb], &output[ibnkb], &parents[ibnkb], b, n);
      for (int j = 0; j < blocks; j++) {
        if (j == k) continue;
        int ibnjb = i * b * n + j * b;
        floyd_warshall_in_place_float(&output[ibnjb], &output[ibnkb],
                                      &output[k * b * n + j * b], &parents[ibnjb], b, n);
      }
    }
  }
}
