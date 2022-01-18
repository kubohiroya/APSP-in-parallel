#pragma once

#include <iostream> // cout
#include <string> // string
#include <sstream> // stringstream
#include <chrono> // see Timer class
#include <random> // mt19937_64, uniform_x_distribution

#include "getInf.hpp"

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
template<typename Number> Number* create_random_adjacencyMatrix(const int n, const double p, const unsigned long seed);

// we need this to initialized to 0 on the diagonal, infinity anywhere there is no edge
// we also need to limit the width and height but keep it a multiple of block_size
// template<typename Number> Number* floyd_warshall_blocked_random_init(const int n, const int block_size, const double p, const unsigned long seed);

template<typename Number> Number* create_random_adjacencyMatrix(const int n, const double p, const unsigned long seed) {
  static const Number inf = getInf<Number>();
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_real_distribution<double> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);
  Number *out = new Number[n * n];
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

template<typename Number> inline void print(const Number value) {
  Number inf = getInf<Number>();
  if (value == inf) {
    std::cout << "Inf";
  } else {
    std::cout << value;
  }
}

template<typename Number> inline void print_matrix(const Number *distanceMatrix, const int n_distanceMatrix, const int n_blocked) {
  for (int i = 0; i < n_distanceMatrix; i++) {
    print<Number>(distanceMatrix[i * n_blocked]);
    for (int j = 1; j < n_distanceMatrix; j++) {
      std::cout << ", ";
      print<Number>(distanceMatrix[i * n_blocked + j]);
    }
    std::cout << std::endl;
  }
}

template<typename Number> inline bool correctness_check(const Number *distanceMatrix, int n_distanceMatrix, const Number *solution, int n_solution) {
  for (int i = 0; i < n_solution; i++) {
    for (int j = 0; j < n_solution; j++) {
      if (distanceMatrix[i * n_distanceMatrix + j] != solution[i * n_solution + j]) {
        std::cerr << "\nAdjacencyMatrix did not match at [" << i << "][" << j << "]: " << distanceMatrix[i * n_distanceMatrix + j]
                  << " vs solution's " << solution[i * n_solution + j] << "!" << std::endl;
        return false;
      }
    }
  }
  return true;
}

inline void print_usage() {
  std::cout << "\nUsage: apsp [-n INT] [-p DOUBLE] [-a (f|j)] [-s LONG] [-b] [-c] [-t INT] [-d INT] [-T (i|f|d)]\n";

  std::cout << "\t-h\t\tPrint this message\n";
  std::cout << "\t-n INT\t\tGraph size, default 1024\n";
  std::cout << "\t-p DOUBLE\tProbability of edge from a given node to another (0.0 to 1.0), default 0.5\n";
  std::cout << "\t-a CHAR\t\tAlgorithm to use for all pairs shortest path\n";
  std::cout << "\t\t\t\tf: Floyd-Warshall (default)\n";
  std::cout << "\t\t\t\tj: Johnson's Algorithm\n";
  std::cout << "\t-s LONG\t\tSeed for graph generation\n";
  std::cout << "\t-b\t\tRun benchmark sequential vs parallel\n";
  std::cout << "\t-t INT\t\tNumber of threads to run\n";
  std::cout << "\t-d INT\t\tBlock size for Blocked-Floyd-Warshall\n";
  std::cout << "\t-c\t\tCheck correctness\n";
  std::cout << "\t-T CHAR\t\tweight type of edge default i\n";
  std::cout << "\n";
}

inline void print_table_row(double p, int v, double seq, double par, bool correct, bool check_correctness) {
  std::printf("\n| %-1.5f | %7d | %12.3f | %12.3f | %10.3f |", p, v, seq, par, seq / par);
  if (check_correctness) {
    std::printf(" %7s |", (correct ? "OK" : "NG"));
  }
}

inline void print_table_break(bool check_correctness) {
  if (check_correctness) {
    std::printf("\n -------------------------------------------------------------------------");
  } else {
    std::printf("\n --------------------------------------------------------------");
  }
}

inline void print_table_header(bool check_correctness) {
  print_table_break(check_correctness);
  std::printf("\n| %-7s | %-7s | %-12s | %-12s | %-10s |",
              "p", "verts", "seq (ms)", "par (ms)", "speedup");
  if (check_correctness) {
    std::printf(" %-7s |", "correct");
  }
  print_table_break(check_correctness);
}

inline std::string get_solution_filename(std::string prefix, int n, double p, unsigned long seed, char typeChar) {
  std::stringstream solution_filename;
  solution_filename << "solution_cache/" << prefix << "-sol-n" << n << "-p" << p << "-s" << seed << "-T" << typeChar
                    << ".bin";
  return solution_filename.str();
}

class Timer {

public:
  Timer() {
    start = std::chrono::high_resolution_clock::now();
  };

  ~Timer() {
    std::chrono::time_point <std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> start_to_end = end - start;
    std::cout << "Timer runtime: " << start_to_end.count() << "\n\n";
  };

private:
  std::chrono::time_point <std::chrono::high_resolution_clock> start;

};
