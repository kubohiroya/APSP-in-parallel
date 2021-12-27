#include "floyd_warshall.hpp"

void floyd_warshall_blocked_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int b, const int n){
  floyd_warshall_blocked<double>(adjacencyMatrix, distanceMatrix, successorMatrix, b, n, DBL_INF);
}
void free_floyd_warshall_blocked_double(double **distanceMatrix, int **successorMatrix){
  free_floyd_warshall_blocked<double>(distanceMatrix, successorMatrix);
}
void floyd_warshall_blocked_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int b, const int n){
  floyd_warshall_blocked<float>(adjacencyMatrix, distanceMatrix, successorMatrix, b, n, FLT_INF);
}
void free_floyd_warshall_blocked_float(float **distanceMatrix, int **successorMatrix){
  free_floyd_warshall_blocked<float>(distanceMatrix, successorMatrix);
}
void floyd_warshall_blocked_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int b, const int n){
  floyd_warshall_blocked<int>(adjacencyMatrix, distanceMatrix, successorMatrix, b, n, INT_INF);
}
void free_floyd_warshall_blocked_int(int **distanceMatrix, int **successorMatrix){
  free_floyd_warshall_blocked<int>(distanceMatrix, successorMatrix);
}
