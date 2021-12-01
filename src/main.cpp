#include <unistd.h> // getopt
#include <cstring> // memcpy
#include <iostream> // cout

#include "main_int.hpp"
#include "main_float.hpp"
#include "main_double.hpp"
#include "util.hpp"

int main(int argc, char *argv[]) {
  // parameter defaults
  unsigned long seed = 0;
  int n = 1024;
  double p = 0.01;
  bool use_floyd_warshall = true;
  bool benchmark = false;
  bool check_correctness = false;
  int block_size = 32;
  int thread_count = 1;
  int type = 0;
  extern char *optarg;
  int opt;
  while ((opt = getopt(argc, argv, "ha:n:p:s:bd:ct:T:")) != -1) {
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

      case 'd':
        block_size = std::stoi(optarg);
        break;

      case 'c':
        check_correctness = true;
        break;

      case 't':
        thread_count = std::stoi(optarg);
        break;

      case 'T':
        if (optarg[0] == 'i') {
          type = 0; // int
        } else if (optarg[0] == 'f') {
          type = 1; // float
        } else if (optarg[0] == 'd') {
          type = 2; // double
        } else {
          std::cerr << "Illegal type argument (neigher i, f nor d)\n";
          return -1;
        }
        break;
    }
  }

  if (type == 0) {
    return do_main_int(seed, n, p, use_floyd_warshall, benchmark, check_correctness, block_size, thread_count);
  } else if (type == 1) {
    return do_main_float(seed, n, p, use_floyd_warshall, benchmark, check_correctness, block_size, thread_count);
  } else if (type == 2) {
    return do_main_double(seed, n, p, use_floyd_warshall, benchmark, check_correctness, block_size, thread_count);
  } else {
    return -1;
  }
}

