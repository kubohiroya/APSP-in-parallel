#include <sys/stat.h> // stat
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
#include "main_double.hpp"

int do_main_double(
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

  double *solution = nullptr; // both algorithms share the same solution
  if (!benchmark && check_correctness) {
    bool write_solution_to_file = true;

    // have we cached the solution before?
    std::string solution_filename = get_solution_filename("apsp", n, p, seed, 'd');
    struct stat file_stat;
    bool solution_available = stat(solution_filename.c_str(), &file_stat) != -1 || errno != ENOENT;

    solution = new double[n * n];
    if (solution_available) {
      std::cout << "Reading reference solution from file: " << solution_filename << "\n";

      std::ifstream in(solution_filename, std::ios::in | std::ios::binary);
      in.read(reinterpret_cast<char *>(solution), n * n * sizeof(double));
      in.close();
    } else {
      double *matrix = floyd_warshall_random_init<double>(n, p, seed, DBL_INF);
      memcpy(solution, matrix, sizeof(double) * n * n);
      int *successorMatrix = new int[n * n];
      auto start = std::chrono::high_resolution_clock::now();
      floyd_warshall<double>(solution, successorMatrix, n);
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
        out.write(reinterpret_cast<const char *>(solution), n * n * sizeof(double));
        out.close();
      }

      delete[] matrix;
      delete[] successorMatrix;
    }
  }

  if (use_floyd_warshall) {
    double *matrix = nullptr;
    double *distanceMatrix = nullptr;
    int *successorMatrix = nullptr;

    if (benchmark) {
      bench_floyd_warshall_double(1, seed, block_size, check_correctness);
    } else {
      matrix = floyd_warshall_blocked_random_init<double>(n, block_size, p, seed, DBL_INF);
      int n_blocked = n;
      int block_remainder = n % block_size;
      if (block_remainder != 0) {
        n_blocked = n + block_size - block_remainder;
      }
      distanceMatrix = new double[n_blocked * n_blocked];
      successorMatrix = new int[n_blocked * n_blocked];

      std::cout << "Using Floyd-Warshall's on " << n_blocked << "x" << n_blocked
                << " with p=" << p << " and seed=" << seed << "\n";
      auto start = std::chrono::high_resolution_clock::now();
#ifdef CUDA
      floyd_warshall_blocked_cuda_double(matrix, &distanceMatrix, &successorMatrix, n_blocked);
#else
      floyd_warshall_blocked_double(matrix, &distanceMatrix, &successorMatrix, n_blocked, block_size);
#endif
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (check_correctness) {
        correctness_check<double>(distanceMatrix, n_blocked, solution, n);
        std::cout << "[matrix]\n";
        print_matrix<double>(matrix, n, n_blocked, DBL_INF);
        std::cout << "[distanceMatrix]\n";
        print_matrix<double>(distanceMatrix, n, n_blocked, DBL_INF);
        std::cout << "[solution]\n";
        print_matrix<double>(solution, n, n, DBL_INF);
        std::cout << "[successorMatrix]\n";
        print_matrix<int>(successorMatrix, n, n_blocked, INT_INF);
      }
      delete[] matrix;
      delete[] distanceMatrix;
      delete[] successorMatrix;
    }
  } else {  // Using Johnson's Algorithm
    if (benchmark) {
      bench_johnson_double(1, seed, check_correctness);
    } else {
      double *distanceMatrix = new double[n * n];
      int *successorMatrix = new int[n * n];

      std::cout << "Using Johnson's on " << n << "x" << n
                << " with p=" << p << " and seed=" << seed << "\n";
#ifdef CUDA
      std::cout << "CUDA!\n";
      graph_cuda_t<double> * cuda_gr = johnson_cuda_random_init<double>(n, p, seed, DBL_INF);
      auto start = std::chrono::high_resolution_clock::now();
      johnson_cuda_double(cuda_gr, distanceMatrix, successorMatrix, DBL_INF);
      auto end = std::chrono::high_resolution_clock::now();
      free_cuda_graph<double>(cuda_gr);
#else
      graph_t<double> * gr = init_random_graph<double>(n, p, seed, DBL_INF);
      auto start = std::chrono::high_resolution_clock::now();
      johnson_parallel<double>(gr, distanceMatrix, successorMatrix, DBL_INF);
      auto end = std::chrono::high_resolution_clock::now();
      free_graph<double>(gr);
#endif
      std::chrono::duration<double, std::milli> start_to_end = end - start;
      std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n\n";

      if (check_correctness) {
        correctness_check<double>(distanceMatrix, n, solution, n);
        std::cout << "[distanceMatrix]\n";
        print_matrix<double>(distanceMatrix, n, n, DBL_INF);
        std::cout << "[solution]\n";
        print_matrix<double>(solution, n, n, DBL_INF);
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

void bench_floyd_warshall_double(int iterations, unsigned long seed, int block_size, bool check_correctness) {
  std::cout << "\n\nFloyd-Warshall's Algorithm benchmarking results for seed=" << seed << " and block size="
            << block_size << "\n";

  print_table_header(check_correctness);
  for (double p = 0.25; p < 1.0; p += 0.25) {
    for (int v = 64; v <= 1024; v *= 2) {
      double *matrix = floyd_warshall_random_init<double>(v, p, seed, DBL_INF);
      double *solution = new double[v * v];

      double *matrix_blocked = matrix; // try to reuse adjacencyMatrixs
      int v_blocked = v;
      int block_remainder = v % block_size;
      if (block_remainder != 0) {
        // we may have to add some verts to fit to a multiple of block_size
        matrix_blocked = floyd_warshall_blocked_random_init<double>(v, block_size, p, seed, DBL_INF);
        v_blocked = v + block_size - block_remainder;
      }
      double *distanceMatrix = new double[v_blocked * v_blocked];
      int *successorMatrix = new int[v_blocked * v_blocked];

      bool correct = false;

      double seq_total_time = 0.0;
      double total_time = 0.0;
      for (int b = 0; b < iterations; b++) {
        // clear solution
        std::memcpy(solution, matrix, v * v * sizeof(double));

        auto seq_start = std::chrono::high_resolution_clock::now();
        floyd_warshall<double>(solution, successorMatrix, v);
        auto seq_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
        seq_total_time += seq_start_to_end.count();

        // clear distanceMatrix
        std::memset(distanceMatrix, 0, v_blocked * v_blocked * sizeof(double));
        auto start = std::chrono::high_resolution_clock::now();
        floyd_warshall_blocked<double>(matrix_blocked, &distanceMatrix, &successorMatrix, v_blocked, block_size, DBL_INF);
        auto end = std::chrono::high_resolution_clock::now();

        if (check_correctness) {
          correct = correct || correctness_check<double>(distanceMatrix, v_blocked, solution, v);
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

void bench_johnson_double(int iterations, unsigned long seed, bool check_correctness) {
  std::cout << "\n\nJohnson's Algorithm benchmarking results for seed=" << seed << "\n";

  print_table_header(check_correctness);
  for (double pp = 0.25; pp < 1.0; pp += 0.25) {
    for (int v = 64; v <= 2048; v *= 2) {
      // johnson init
      graph_t<double> *gr = init_random_graph<double>(v, pp, seed, DBL_INF);
      double *matrix = floyd_warshall_random_init<double>(v, pp, seed, DBL_INF);
      double *distanceMatrix = new double[v * v];
      int *successorMatrix = new int[v * v];

      double *solution = new double[v * v];
      double **out_sol = new double *[v];
      for (int i = 0; i < v; i++) out_sol[i] = &solution[i * v];
      Graph<double> G(gr->edge_array, gr->edge_array + gr->E, gr->weights, gr->V);
      std::vector<double> d(num_vertices(G));
      std::vector<int> p(num_vertices(G));

      bool correct = false;

      double seq_total_time = 0.0;
      double total_time = 0.0;
      for (int b = 0; b < iterations; b++) {
        // clear solution
        std::memset(solution, 0, v * v * sizeof(double));

        auto seq_start = std::chrono::high_resolution_clock::now();
        johnson_all_pairs_shortest_paths(G, out_sol, distance_map(&d[0]).predecessor_map(&p[0]));
        auto seq_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
        seq_total_time += seq_start_to_end.count();

        // clear distanceMatrix
        std::memset(distanceMatrix, 0, v * v * sizeof(double));
        std::memset(successorMatrix, 0, v * v * sizeof(int));
        auto start = std::chrono::high_resolution_clock::now();
        // TODO: johnson parallel -- temporarily putting floyd_warshall here
        //floyd_warshall_blocked(matrix, distanceMatrix, v, block_size);
        johnson_parallel<double>(gr, distanceMatrix, successorMatrix, DBL_INF);
        auto end = std::chrono::high_resolution_clock::now();

        if (check_correctness) {
          correct = correct || correctness_check<double>(distanceMatrix, v, solution, v);
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
