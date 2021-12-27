#include <sys/stat.h> // stat
#include <unistd.h> // getopt
#include <chrono> // high_resolution_clock
#include <iostream> // cout
#include <cstdio> // printf
#include <fstream> // ifstream, ofstream
#include <sstream> // stringstream
#include <ratio>  // milli

#ifdef _OPENMP
#include "omp.h" // omp_set_num_threads
#endif

#include <boost/graph/johnson_all_pairs_shortest.hpp> // seq algorithm, distance_map

#include "inf.hpp"
#include "util.hpp"
#include "floyd_warshall.hpp"
#include "johnson.hpp"
#include "main_float.hpp"

int do_main_float(
    unsigned long seed,
    int n,
    double p,
    bool use_floyd_warshall,
    bool benchmark,
    bool check_correctness,
    int block_size,
    int thread_count
) {

#ifdef _OPENMP
  omp_set_num_threads(thread_count);
#else
  (void) thread_count; // suppress unused warning
#endif

  float *solution = nullptr; // both algorithms share the same solution
  if (!benchmark && check_correctness) {
    bool write_solution_to_file = true;

    // have we cached the solution before?
    std::string solution_filename = get_solution_filename("apsp", n, p, seed, 'f');
    struct stat file_stat;
    bool solution_available = stat(solution_filename.c_str(), &file_stat) != -1 || errno != ENOENT;

    solution = new float[n * n];
    if (solution_available) {
      std::cout << "Reading reference solution from file: " << solution_filename << "\n";

      std::ifstream in(solution_filename, std::ios::in | std::ios::binary);
      in.read(reinterpret_cast<char *>(solution), n * n * sizeof(float));
      in.close();
    } else {
      float *matrix = floyd_warshall_random_init<float>(n, p, seed, FLT_INF);
      memcpy(solution, matrix, sizeof(float) * n * n);
      int *successorMatrix = new int[n * n];
      auto start = std::chrono::high_resolution_clock::now();
      floyd_warshall<float>(solution, successorMatrix, n);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n";

      if (write_solution_to_file) {
        std::cout << "Writing solution to file: " << solution_filename << "\n";

        if (system("mkdir -p solution_cache") == -1) {
          std::cerr << "mkdir failed!";
          return -1;
        }

        std::ofstream out(solution_filename, std::ios::out | std::ios::binary);
        out.write(reinterpret_cast<const char *>(solution), n * n * sizeof(float));
        out.close();
      }

      delete[] matrix;
      delete[] successorMatrix;
    }
  }

  if (use_floyd_warshall) {
    float *matrix = nullptr;
    float *distanceMatrix = nullptr;
    int *successorMatrix = nullptr;

    if (benchmark) {
      bench_floyd_warshall_float(1, seed, block_size, check_correctness);
    } else {
      matrix = floyd_warshall_blocked_random_init<float>(n, block_size, p, seed, FLT_INF);
      int n_blocked = n;
      int block_remainder = n % block_size;
      if (block_remainder != 0) {
        n_blocked = n + block_size - block_remainder;
      }
      distanceMatrix = new float[n_blocked * n_blocked];
      successorMatrix = new int[n_blocked * n_blocked];

      std::cout << "Using Floyd-Warshall's on " << n_blocked << "x" << n_blocked
                << " with p=" << p << " and seed=" << seed << "\n";
      auto start = std::chrono::high_resolution_clock::now();
#ifdef CUDA
      floyd_warshall_blocked_cuda_float(matrix, &distanceMatrix, &successorMatrix, n_blocked);
#else
      floyd_warshall_blocked_float(matrix, &distanceMatrix, &successorMatrix, n_blocked, block_size);
#endif
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (check_correctness) {
        correctness_check<float>(distanceMatrix, n_blocked, solution, n);

        std::cout << "[matrix]\n";
        print_matrix<float>(matrix, n, n_blocked, FLT_INF);
        std::cout << "[distanceMatrix]\n";
        print_matrix<float>(distanceMatrix, n, n_blocked, FLT_INF);
        std::cout << "[solution]\n";
        print_matrix<float>(solution, n, n, FLT_INF);
        std::cout << "[successorMatrix]\n";
        print_matrix<int>(successorMatrix, n, n_blocked, INT_INF);
      }
      delete[] matrix;
      delete[] distanceMatrix;
      delete[] successorMatrix;
    }
  } else {  // Using Johnson's Algorithm
    if (benchmark) {
      bench_johnson_float(1, seed, check_correctness);
    } else {
      float *distanceMatrix = new float[n * n];
      int *successorMatrix = new int[n * n];
      std::cout << "Using Johnson's on " << n << "x" << n
                << " with p=" << p << " and seed=" << seed << "\n";
#ifdef CUDA
      std::cout << "CUDA!\n";
      graph_cuda_t<float>* cuda_gr = johnson_cuda_random_init<float>(n, p, seed, FLT_INF);
      auto start = std::chrono::high_resolution_clock::now();
      johnson_cuda<float>(cuda_gr, distanceMatrix, successorMatrix);
      auto end = std::chrono::high_resolution_clock::now();
      free_cuda_graph<float>(cuda_gr);
#else
      graph_t<float> *gr = init_random_graph<float>(n, p, seed, FLT_INF);
      auto start = std::chrono::high_resolution_clock::now();
      johnson_parallel<float>(gr, distanceMatrix, successorMatrix, FLT_INF);
      auto end = std::chrono::high_resolution_clock::now();
      free_graph<float>(gr);
#endif
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (check_correctness) {
        correctness_check<float>(distanceMatrix, n, solution, n);
        std::cout << "[distanceMatrix]\n";
        print_matrix<float>(distanceMatrix, n, n, FLT_INF);
        std::cout << "[solution]\n";
        print_matrix<float>(solution, n, n, FLT_INF);
        std::cout << "[successorMatrix]\n";
        print_matrix<int>(successorMatrix, n, n, INT_INF);
      }

      delete[] distanceMatrix;
      delete[] successorMatrix;
    }
  }

  if (check_correctness) {
    delete[] solution;
  }

  return 0;
}

void bench_floyd_warshall_float(int iterations, unsigned long seed, int block_size, bool check_correctness) {
  std::cout << "\n\nFloyd-Warshall's Algorithm benchmarking results for seed=" << seed << " and block size="
            << block_size << "\n";

  print_table_header(check_correctness);
  for (double p = 0.25; p < 1.0; p += 0.25) {
    for (int v = 64; v <= 1024; v *= 2) {
      float *matrix = floyd_warshall_random_init<float>(v, p, seed, FLT_INF);
      float *solution = new float[v * v];

      float *matrix_blocked = matrix; // try to reuse adjacencyMatrixs
      int v_blocked = v;
      int block_remainder = v % block_size;
      if (block_remainder != 0) {
        // we may have to add some verts to fit to a multiple of block_size
        matrix_blocked = floyd_warshall_blocked_random_init<float>(v, block_size, p, seed, FLT_INF);
        v_blocked = v + block_size - block_remainder;
      }
      float *distanceMatrix = new float[v_blocked * v_blocked];
      int *successorMatrix = new int[v_blocked * v_blocked];

      bool correct = false;

      double seq_total_time = 0.0;
      double total_time = 0.0;
      for (int b = 0; b < iterations; b++) {
        // clear solution
        std::memcpy(solution, matrix, v * v * sizeof(float));

        auto seq_start = std::chrono::high_resolution_clock::now();
        floyd_warshall<float>(solution, successorMatrix, v);
        auto seq_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
        seq_total_time += seq_start_to_end.count();

        // clear distanceMatrix
        std::memset(distanceMatrix, 0, v_blocked * v_blocked * sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        floyd_warshall_blocked_float(matrix_blocked, &distanceMatrix, &successorMatrix, v_blocked, block_size);
        auto end = std::chrono::high_resolution_clock::now();

        if (check_correctness) {
          correct = correct || correctness_check<float>(distanceMatrix, v_blocked, solution, v);
        }

        std::chrono::duration<double, std::milli> start_to_end = end - start;
        total_time += start_to_end.count();
      }
      delete[] matrix;
      delete[] distanceMatrix;
      delete[] successorMatrix;
      delete[] solution;
      print_table_row(p, v, seq_total_time, total_time, check_correctness, correct);
    }
    print_table_break(check_correctness);
  }
  std::cout << "\n\n";
}


void bench_johnson_float(int iterations, unsigned long seed, bool check_correctness) {
  std::cout << "\n\nJohnson's Algorithm benchmarking results for seed=" << seed << "\n";

  print_table_header(check_correctness);
  for (double pp = 0.25; pp < 1.0; pp += 0.25) {
    for (int v = 64; v <= 2048; v *= 2) {
      // johnson init
      graph_t<float> *gr = init_random_graph<float>(v, pp, seed, FLT_INF);
      float *matrix = floyd_warshall_random_init<float>(v, pp, seed, FLT_INF);
      float *distanceMatrix = new float[v * v];
      int *successorMatrix = new int[v * v];

      float *solution = new float[v * v];
      float **out_sol = new float *[v];
      for (int i = 0; i < v; i++) out_sol[i] = &solution[i * v];
      Graph<float> G(gr->edge_array, gr->edge_array + gr->E, gr->weights, gr->V);
      std::vector<float> d(num_vertices(G));
      std::vector<int> p(num_vertices(G));

      bool correct = false;

      double seq_total_time = 0.0;
      double total_time = 0.0;
      for (int b = 0; b < iterations; b++) {
        // clear solution
        std::memset(solution, 0, v * v * sizeof(float));

        auto seq_start = std::chrono::high_resolution_clock::now();
        johnson_all_pairs_shortest_paths(G, out_sol, distance_map(&d[0]).predecessor_map(&p[0]));
        auto seq_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
        seq_total_time += seq_start_to_end.count();

        // clear distanceMatrix
        std::memset(distanceMatrix, 0, v * v * sizeof(float));
        std::memset(successorMatrix, 0, v * v * sizeof(int));
        auto start = std::chrono::high_resolution_clock::now();
        // TODO: johnson parallel -- temporarily putting floyd_warshall here
        //floyd_warshall_blocked(matrix, distanceMatrix, v, block_size);
        johnson_parallel<float>(gr, distanceMatrix, successorMatrix, FLT_INF);
        auto end = std::chrono::high_resolution_clock::now();

        if (check_correctness) {
          correct = correct || correctness_check<float>(distanceMatrix, v, solution, v);
        }

        std::chrono::duration<double, std::milli> start_to_end = end - start;
        total_time += start_to_end.count();
      }
      delete[] solution;
      delete[] out_sol;
      delete[] distanceMatrix;
      delete[] successorMatrix;
      delete[] matrix;

      print_table_row(pp, v, seq_total_time, total_time, check_correctness, correct);
    }
    print_table_break(check_correctness);
  }
  std::cout << "\n\n";
}
