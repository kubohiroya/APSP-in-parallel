#pragma once

#include <iostream> // cout
#include <string> // string
#include <sstream> // stringstream
#include <chrono> // see Timer class

#include "inf.hpp"
#include "equals.hpp"

inline void print_int(const int value) {
  if (value == INT_INF) {
    std::cout << "Inf";
  } else {
    std::cout << value;
  }
}

inline void print_float(const float value) {
  if (equals_float(value, FLT_INF)) {
    std::cout << "Inf";
  } else {
    std::cout << value;
  }
}

inline void print_double(const double value) {
  if (equals_double(value, DBL_INF)) {
    std::cout << "Inf";
  } else {
    std::cout << value;
  }
}

inline void print_matrix_int(const int *distanceMatrix, const int n_distanceMatrix, const int n_blocked) {
  for (int i = 0; i < n_distanceMatrix; i++) {
    print_int(distanceMatrix[i * n_blocked]);
    for (int j = 1; j < n_distanceMatrix; j++) {
      std::cout << ", ";
      print_int(distanceMatrix[i * n_blocked + j]);
    }
    std::cout << std::endl;
  }
}

inline void print_matrix_float(const float *distanceMatrix, const int n_distanceMatrix, const int n_blocked) {
  for (int i = 0; i < n_distanceMatrix; i++) {
    print_float(distanceMatrix[i * n_blocked]);
    for (int j = 1; j < n_distanceMatrix; j++) {
      std::cout << ", ";
      print_float(distanceMatrix[i * n_blocked + j]);
    }
    std::cout << std::endl;
  }
}

inline void print_matrix_double(const double *distanceMatrix, const int n_distanceMatrix, const int n_blocked) {
  for (int i = 0; i < n_distanceMatrix; i++) {
    print_double(distanceMatrix[i * n_blocked]);
    for (int j = 1; j < n_distanceMatrix; j++) {
      std::cout << ", ";
      print_double(distanceMatrix[i * n_blocked + j]);
    }
    std::cout << std::endl;
  }
}

inline bool correctness_check_int(int *distanceMatrix, int n_distanceMatrix, int *solution, int n_solution) {
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

inline bool correctness_check_float(float *distanceMatrix, int n_distanceMatrix, float *solution, int n_solution) {
  for (int i = 0; i < n_solution; i++) {
    for (int j = 0; j < n_solution; j++) {
      if (!equals_float(distanceMatrix[i * n_distanceMatrix + j], solution[i * n_solution + j])) {
        std::cerr << "\nAdjacencyMatrix did not match at [" << i << "][" << j << "]: " << distanceMatrix[i * n_distanceMatrix + j]
                  << " vs solution's " << solution[i * n_solution + j] << "!" << std::endl;
        return false;
      }
    }
  }

  return true;
}

inline bool correctness_check_double(double *distanceMatrix, int n_distanceMatrix, double *solution, int n_solution) {
  for (int i = 0; i < n_solution; i++) {
    for (int j = 0; j < n_solution; j++) {
      if (!equals_double(distanceMatrix[i * n_distanceMatrix + j], solution[i * n_solution + j])) {
        std::cerr << "\nAdjacencyMatrix did not match at [" << i << "][" << j << "]: " << distanceMatrix[i * n_distanceMatrix + j]
                  << " vs solution's " << solution[i * n_solution + j] << "!" << std::endl;
        return false;
      }
    }
  }

  return true;
}

inline void print_usage() {
  std::cout << "\nUsage: apsp [-n INT] [-p DOUBLE] [-a (f|j)] [-s LONG] [-b] [-c] [-t INT] [-T (i|f|d)]\n";

  std::cout << "\t-h\t\tPrint this message\n";
  std::cout << "\t-n INT\t\tGraph size, default 1024\n";
  std::cout << "\t-p DOUBLE\tProbability of edge from a given node to another (0.0 to 1.0), default 0.5\n";
  std::cout << "\t-a CHAR\t\tAlgorithm to use for all pairs shortest path\n";
  std::cout << "\t\t\t\tf: Floyd-Warshall (default)\n";
  std::cout << "\t\t\t\tj: Johnson's Algorithm\n";
  std::cout << "\t-s LONG\t\tSeed for graph generation\n";
  std::cout << "\t-b\t\tRun benchmark sequential vs parallel\n";
  std::cout << "\t-t INT\t\tNumber of threads to run\n";
  std::cout << "\t-c\t\tCheck correctness\n";
  std::cout << "\t-T CHAR\t\tweight type of edge default i\n";
  std::cout << "\n";
}

inline void print_table_row(double p, int v, double seq, double par, bool check_correctness, bool correct) {
  std::printf("\n| %-3.2f | %-7d | %-12.3f | %-12.3f | %-10.3f |", p, v, seq, par, seq / par);
  if (check_correctness) {
    std::printf(" %-8s |", (correct ? "x" : "!"));
  }
}

inline void print_table_break(bool check_correctness) {
  if (check_correctness) {
    std::printf("\n ----------------------------------------------------------------------");
  } else {
    std::printf("\n -----------------------------------------------------------");
  }
}

inline void print_table_header(bool check_correctness) {
  print_table_break(check_correctness);
  std::printf("\n| %-4s | %-7s | %-12s | %-12s | %-10s |",
              "p", "verts", "seq (ms)", "par (ms)", "speedup");
  if (check_correctness) {
    std::printf(" %-8s |", "correct");
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
