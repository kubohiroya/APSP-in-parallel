#include "floyd_warshall.hpp"

void floyd_warshall_blocked_double(const double *adjacencyMatrix, double **distanceMatrix, const int n, const int b){
  floyd_warshall_blocked<double>(adjacencyMatrix, distanceMatrix, b, n);
}
void free_floyd_warshall_blocked_double(double **distanceMatrix){
  free_floyd_warshall_blocked<double>(distanceMatrix);
}
void floyd_warshall_blocked_float(const float *adjacencyMatrix, float **distanceMatrix, const int n, const int b){
  floyd_warshall_blocked<float>(adjacencyMatrix, distanceMatrix, b, n);
}
void free_floyd_warshall_blocked_float(float **distanceMatrix){
  free_floyd_warshall_blocked<float>(distanceMatrix);
}
void floyd_warshall_blocked_int(const int *adjacencyMatrix, int **distanceMatrix, const int n, const int b){
  floyd_warshall_blocked<int>(adjacencyMatrix, distanceMatrix, b, n);
}
void free_floyd_warshall_blocked_int(int **distanceMatrix){
  free_floyd_warshall_blocked<int>(distanceMatrix);
}

void floyd_warshall_blocked_successor_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int n, const int b){
  floyd_warshall_blocked<double>(adjacencyMatrix, distanceMatrix, successorMatrix, b, n);
}
void free_floyd_warshall_blocked_successor_double(double **distanceMatrix, int **successorMatrix){
  free_floyd_warshall_blocked<double>(distanceMatrix, successorMatrix);
}
void floyd_warshall_blocked_successor_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int n, const int b){
  floyd_warshall_blocked<float>(adjacencyMatrix, distanceMatrix, successorMatrix, b, n);
}
void free_floyd_warshall_blocked_successor_float(float **distanceMatrix, int **successorMatrix){
  free_floyd_warshall_blocked<float>(distanceMatrix, successorMatrix);
}
void floyd_warshall_blocked_successor_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int n, const int b){
  floyd_warshall_blocked<int>(adjacencyMatrix, distanceMatrix, successorMatrix, b, n);
}
void free_floyd_warshall_blocked_successor_int(int **distanceMatrix, int **successorMatrix){
  free_floyd_warshall_blocked<int>(distanceMatrix, successorMatrix);
}

