#include <unistd.h> // getopt
#include <cstring> // memcpy
#include <iostream> // cout
#include <sys/stat.h> // stat
#include <chrono> // high_resolution_clock
#include <cstdio> // printf
#include <fstream> // ifstream, ofstream
#include <sstream> // stringstream
#include <ratio>  // milli

#ifdef _OPENMP
#include "omp.h" // omp_set_num_threads
#endif

#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include "inf.hpp"
#include "util.hpp"
#include "floyd_warshall.hpp"
#include "johnson.hpp"

struct bench_result{
  bool correct;
  double seq_total_time;
  double total_time;
};

template<typename Number>
void print_matrix(int n, int n_blocked, Number *adjacencyMatrix, Number *distanceMatrix, Number *solution, int *successorMatrix){
  if(adjacencyMatrix != nullptr){
    std::cout << "[adjacencyMatrix]\n";
    print_matrix<Number>(adjacencyMatrix, n, n_blocked);
  }
  if(distanceMatrix != nullptr){
    std::cout << "[distanceMatrix]\n";
    print_matrix<Number>(distanceMatrix, n, n_blocked);
  }
  if (solution != nullptr) {
    std::cout << "[solution]\n";
    print_matrix<Number>(solution, n, n);
  }
  if (successorMatrix != nullptr) {
    std::cout << "[successorMatrix]\n";
    print_matrix<int>(successorMatrix, n, n_blocked);
  }
}

template<typename Number> inline char getTypeChar();
template<> inline char getTypeChar<double>(){
  return 'd';
}
template<> inline char getTypeChar<float>(){
  return 'f';
}
template<> inline char getTypeChar<int>(){
  return 'i';
}

template<typename Number>
Number* get_solution(
  const int n,
  double p,
  unsigned long seed){
  size_t size = n * n * sizeof(Number);
  Number *solution = (Number *) malloc(size);
  bool write_solution_to_file = true;
  char chr = getTypeChar<Number>();

  // have we cached the solution before?
  std::string solution_filename = get_solution_filename("apsp", n, p, seed, chr);
  struct stat file_stat;
  bool solution_available = stat(solution_filename.c_str(), &file_stat) != -1 || errno != ENOENT;

  if (solution_available) {
    // std::cout << "Reading reference solution from file: " << solution_filename << "\n";
    std::ifstream in(solution_filename, std::ios::in | std::ios::binary);
    in.read(reinterpret_cast<char *>(solution), size);
    in.close();
    // std::cout << "filename: " << solution_filename << " size: " << size << " bytes read.\n";
  } else {
    const Number *adjacencyMatrix = create_random_adjacencyMatrix<Number>(n, p, seed);
    auto start = std::chrono::high_resolution_clock::now();
    if(p > 0.1){
      floyd_warshall_blocked<Number>(adjacencyMatrix, &solution, n, 32);
    }else{
      johnson_parallel_matrix<Number>(adjacencyMatrix, &solution, n);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> start_to_end = end - start;
    std::cout << "Algorithm runtime: " << start_to_end.count() << "ms\n";

    delete[] adjacencyMatrix;

    if (write_solution_to_file) {
      std::cout << "Writing solution to file: " << solution_filename << "\n";

      if (system("mkdir -p solution_cache") == -1) {
        std::cerr << "mkdir failed!";
        return nullptr;
      }

      std::ofstream out(solution_filename, std::ios::out | std::ios::binary);
      out.write(reinterpret_cast<const char *>(solution), size);
      out.close();
    }
  }
  return solution;
}

template<typename Number>
double do_floyd_warshall(int n, int block_size, double p, unsigned long seed, bool with_successor, bool check_correctness){

  Number *solution = check_correctness ? get_solution<Number>(n, p, seed) : nullptr;

  Number *adjacencyMatrix = create_random_adjacencyMatrix<Number>(n, p, seed);
  Number *distanceMatrix = nullptr;
  int *successorMatrix = nullptr;

  auto start = std::chrono::high_resolution_clock::now();
  if (with_successor) {
    floyd_warshall_successor_blocked<Number>(adjacencyMatrix, &distanceMatrix, &successorMatrix, n, block_size);
  } else {
    floyd_warshall_blocked<Number>(adjacencyMatrix, &distanceMatrix, n, block_size);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> start_to_end = end - start;

  if (check_correctness) {
    if(n <= 256) print_matrix(n, n, adjacencyMatrix, distanceMatrix, solution, successorMatrix);
    correctness_check<Number>(distanceMatrix, n, solution, n);
  }
  if (check_correctness) {
    delete[] solution;
  }
  if (with_successor) {
    delete[] successorMatrix;
  }
  delete[] distanceMatrix;
  delete[] adjacencyMatrix;

  return start_to_end.count();
}


template<typename Number>
bench_result * bench_floyd_warshall(int iterations, int v, int block_size, double p, unsigned long seed, bool with_successor, bool check_correctness) {

  Number *solution = check_correctness ? get_solution<Number>(v, p, seed) : nullptr;
  const Number *adjacencyMatrix = create_random_adjacencyMatrix<Number>(v, p, seed);
  static const Number inf = getInf<Number>();

  bench_result *result = new bench_result;
  result->correct = true;
  result->seq_total_time = 0.0;
  result->total_time = 0.0;

  for (int b = 0; b < iterations; b++) {

    Number *distanceMatrix = nullptr;
    int *successorMatrix = nullptr;

    auto seq_start = std::chrono::high_resolution_clock::now();
    if (with_successor) {
      floyd_warshall<Number>(adjacencyMatrix, &distanceMatrix, &successorMatrix, v);
    } else {
      floyd_warshall<Number>(adjacencyMatrix, &distanceMatrix, v);
    }
    auto seq_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
    result->seq_total_time += seq_start_to_end.count() / iterations;;

    delete[] distanceMatrix;
    if(with_successor){
      delete[] successorMatrix;
    }

    distanceMatrix = nullptr;
    successorMatrix = nullptr;

    auto start = std::chrono::high_resolution_clock::now();
    if (with_successor) {
      floyd_warshall_successor_blocked<Number>(adjacencyMatrix, &distanceMatrix, &successorMatrix, v, block_size);
    } else {
      floyd_warshall_blocked<Number>(adjacencyMatrix, &distanceMatrix, v, block_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> start_to_end = end - start;
    result->total_time += start_to_end.count() / iterations;;

    if (check_correctness) {
      result->correct = result->correct && correctness_check<Number>(distanceMatrix, v, solution, v);
    }

    delete[] distanceMatrix;
    if(with_successor){
      delete[] successorMatrix;
    }
  }

  if (check_correctness) {
    delete[] solution;
  }
  delete[] adjacencyMatrix;

  return result;
}

template<typename Number>
double do_johnson(const int n, const double p, const unsigned long seed, const bool with_successor, const bool check_correctness){

  Number *solution = check_correctness ? get_solution<Number>(n, p, seed) : nullptr;
  Number *adjacencyMatrix = create_random_adjacencyMatrix<Number>(n, p, seed);
  Number *distanceMatrix = nullptr;
  int *successorMatrix = nullptr;

  auto start = std::chrono::high_resolution_clock::now();

  if (with_successor) {
    johnson_parallel_matrix<Number>(adjacencyMatrix, &distanceMatrix, &successorMatrix, n);
  } else {
    johnson_parallel_matrix<Number>(adjacencyMatrix, &distanceMatrix, n);
  }

  auto end = std::chrono::high_resolution_clock::now();

  if (check_correctness) {
    if(n <= 256) print_matrix(n, n, adjacencyMatrix, distanceMatrix, solution, successorMatrix);
    correctness_check<Number>(distanceMatrix, n, solution, n);
  }

  if(distanceMatrix != nullptr){
    delete[] distanceMatrix;
  }
  if (with_successor) {
    delete[] successorMatrix;
  }
  if (check_correctness){
    delete[] solution;
  }
  delete[] adjacencyMatrix;

  std::chrono::duration<double, std::milli> start_to_end = end - start;
  return start_to_end.count();
}

template<typename Number>
bench_result * bench_johnson(int iterations, int nvertex, double p, unsigned long seed, bool with_successor, bool check_correctness) {

  Number *solution = check_correctness ? get_solution<Number>(nvertex, p, seed) : nullptr;
  Number *adjacencyMatrix = create_random_adjacencyMatrix<Number>(nvertex, p, seed);

  static const Number inf = getInf<Number>();
  bench_result *result = new bench_result;
  result->correct = true;
  result->seq_total_time = 0.0;
  result->total_time = 0.0;

  for (int b = 0; b < iterations; b++) {
    Number *distanceMatrix = nullptr;
    int *successorMatrix = nullptr;

    auto start = std::chrono::high_resolution_clock::now();
    if (with_successor) {
      johnson_parallel_matrix<Number>(adjacencyMatrix, &distanceMatrix, &successorMatrix, nvertex);
    } else {
      johnson_parallel_matrix<Number>(adjacencyMatrix, &distanceMatrix, nvertex);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> start_to_end = end - start;
;
    result->total_time += start_to_end.count() / iterations;

    if (solution != nullptr) {
      result->correct = result->correct && correctness_check<Number>(distanceMatrix, nvertex, solution, nvertex);
    }

    delete[] distanceMatrix;
    if(with_successor){
      delete[] successorMatrix;
    }

    auto seq_start = std::chrono::high_resolution_clock::now();
    graph_t<Number> *gr = init_graph<Number>(adjacencyMatrix, nvertex);
    Graph<Number> G(gr->edges, gr->edges + gr->E, gr->weights, gr->V);
    std::vector<Number> d(num_vertices(G));
    std::vector<int> predecessor(num_vertices(G));
    Number **distanceArray = new Number *[nvertex];

    for (int i = 0; i < nvertex; i++) distanceArray[i] = &adjacencyMatrix[i * nvertex];
    if (with_successor) {
      johnson_all_pairs_shortest_paths(G, distanceArray, distance_map(&d[0]).predecessor_map(&predecessor[0]).distance_inf(inf));
    }else{
      johnson_all_pairs_shortest_paths(G, distanceArray, distance_map(&d[0]).distance_inf(inf));
    }

    auto seq_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_start_to_end = seq_end - seq_start;
    result->seq_total_time += seq_start_to_end.count() / iterations;

    delete[] distanceArray;
    free_graph<Number>(gr);

  }

  if (check_correctness) {
    delete[] solution;
  }
  delete[] adjacencyMatrix;

  return result;
}

template<typename Number>
int do_benchmark(
  int iterations,
  unsigned long seed,
  bool use_floyd_warshall,
  int block_size,
  bool with_successor,
  bool check_correctness
) {

  if (use_floyd_warshall) {
      std::cout << "\n\nFloyd-Warshall's Algorithm benchmarking results for seed=" << seed << " and block size="
                << block_size << "\n";
      print_table_header(check_correctness);

      for (double p = 0.25; p < 1.0; p += 0.25) {
        for (int v = 64; v <= 4096; v *= 2) {
          bench_result *result = bench_floyd_warshall<Number>(iterations, v, block_size, p, seed, with_successor, check_correctness);
          print_table_row(p, v, result->seq_total_time, result->total_time, result->correct, check_correctness);
          delete result;
        }
        print_table_break(check_correctness);
      }
      std::cout << "\n\n";
  } else {  // Using Johnson's Algorithm
      std::cout << "\n\nJohnson's Algorithm benchmarking results for seed=" << seed << "\n";
      print_table_header(check_correctness);
      for (double p = 0.025; p < 0.1; p += 0.025) {
        for (int v = 64; v <= 4096; v *= 2) {
          bench_result *result = bench_johnson<Number>(iterations, v, p, seed, with_successor, check_correctness);
          print_table_row(p, v, result->seq_total_time, result->total_time, result->correct, check_correctness);
          delete result;
        }
        print_table_break(check_correctness);
      }
      std::cout << "\n\n";
  }

  return 0;
}

template<typename Number>
int do_main(
  int n,
  double p,
  unsigned long seed,
  bool with_successor,
  bool use_floyd_warshall,
  int block_size,
  bool check_correctness
) {

  if (use_floyd_warshall) {
      std::cout << "Using Floyd-Warshall's on " << n << "x" << n
                << " with p=" << p << " and seed=" << seed << "\n";
      double start_to_end_count = do_floyd_warshall<Number>(n, block_size, p, seed, with_successor, check_correctness);
      std::cout << "Algorithm runtime: " << start_to_end_count << "ms\n\n";
  } else {  // Using Johnson's Algorithm
      std::cout << "Using Johnson's on " << n << "x" << n
                << " with p=" << p << " and seed=" << seed << "\n";
      double start_to_end_count = do_johnson<Number>(n, p, seed, with_successor, check_correctness);
      std::cout << "Algorithm runtime: " << start_to_end_count << "ms\n\n";
  }

  return 0;
}


int main(int argc, char *argv[]) {
  // parameter defaults
  unsigned long seed = 0;
  int n = 1024;
  double p = 0.01;
  bool use_floyd_warshall = true;
  bool benchmark = false;
  bool check_correctness = false;
  bool with_successor = false;
  int block_size = 16;
  int thread_count = 1;
  char type = 'i';
  int iterations = 5;

  extern char *optarg;
  int opt;
  while ((opt = getopt(argc, argv, "ha:n:p:s:bd:ct:T:Si:")) != -1) {
    switch (opt) {
      case 'h':
      case '?': // illegal command
      case ':': // forgot command's argument
        print_usage();
        return 0;

      case 'a':
        if (optarg[0] == 'j') {
          use_floyd_warshall = false;
        } else if (optarg[0] != 'f') {
          std::cerr << "Illegal algorithm argument, must be f or j\n";
          return -1;
        }
        break;

      case 'p':
        p = std::stod(optarg);
        break;

      case 's':
        seed = std::stoul(optarg);
        break;

      case 'b':
        benchmark = true;
        break;

      case 'n':
        n = std::stoi(optarg);
        break;

      case 'i':
        iterations = std::stoi(optarg);
        break;

      case 'd':
        block_size = std::stoi(optarg);
        break;

      case 'c':
        check_correctness = true;
        break;

      case 't':
        thread_count = std::stoi(optarg);
        break;

      case 'S':
        with_successor = true;
        break;

      case 'T':
        if (optarg[0] == 'i') {
          type = 'i'; // int
        } else if (optarg[0] == 'f') {
          type = 'f'; // float
        } else if (optarg[0] == 'd') {
          type = 'd'; // double
        } else {
          std::cerr << "Illegal type argument (neigher i, f nor d)\n";
          return -1;
        }
        break;
    }
  }

#ifdef _OPENMP
  if(thread_count > 1){
    omp_set_num_threads(thread_count);
  }
#else
  (void) thread_count; // suppress unused warning
#endif

  if(benchmark){
    switch(type){
      case 'i':
        return do_benchmark<int>(iterations, seed, use_floyd_warshall, block_size, with_successor, check_correctness);
      case 'f':
        return do_benchmark<float>(iterations, seed, use_floyd_warshall, block_size, with_successor, check_correctness);
      case 'd':
        return do_benchmark<double>(iterations, seed, use_floyd_warshall, block_size, with_successor, check_correctness);
    }
  }else{
    switch(type){
      case 'i':
        return do_main<int>(n, p, seed, with_successor, use_floyd_warshall, block_size, check_correctness);
      case 'f':
        return do_main<float>(n, p, seed, with_successor, use_floyd_warshall, block_size, check_correctness);
      case 'd':
        return do_main<double>(n, p, seed, with_successor, use_floyd_warshall, block_size, check_correctness);
    }
  }
  return -1;
}
