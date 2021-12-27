#include <iostream> // cerr
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include "johnson.hpp"
#include "equals.hpp"

void johnson_parallel_matrix_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int n){
  johnson_parallel_matrix<double>(adjacencyMatrix, distanceMatrix, successorMatrix, n, DBL_INF);
}
void free_johnson_parallel_matrix_double(double **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_matrix<double>(distanceMatrix, successorMatrix);
}
void johnson_parallel_matrix_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int n){
  johnson_parallel_matrix<float>(adjacencyMatrix, distanceMatrix, successorMatrix, n, FLT_INF);
}
void free_johnson_parallel_matrix_float(float **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_matrix<float>(distanceMatrix, successorMatrix);
}
void johnson_parallel_matrix_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int n){
  johnson_parallel_matrix<int>(adjacencyMatrix, distanceMatrix, successorMatrix, n, INT_INF);
}
void free_johnson_parallel_matrix_int(int **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_matrix<int>(distanceMatrix, successorMatrix);
}
