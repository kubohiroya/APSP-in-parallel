#include <iostream> // cerr
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include "getInf.hpp"
#include "johnson.hpp"

void johnson_parallel_matrix_double(const double *adjacencyMatrix, double **distanceMatrix, const int n){
  johnson_parallel_matrix<double>(adjacencyMatrix, distanceMatrix, n);
}
void free_johnson_parallel_matrix_double(double **distanceMatrix){
  free_johnson_parallel_matrix<double>(distanceMatrix);
}
void johnson_parallel_matrix_float(const float *adjacencyMatrix, float **distanceMatrix, const int n){
  johnson_parallel_matrix<float>(adjacencyMatrix, distanceMatrix, n);
}
void free_johnson_parallel_matrix_float(float **distanceMatrix){
  free_johnson_parallel_matrix<float>(distanceMatrix);
}
void johnson_parallel_matrix_int(const int *adjacencyMatrix, int **distanceMatrix, const int n){
  johnson_parallel_matrix<int>(adjacencyMatrix, distanceMatrix, n);
}
void free_johnson_parallel_matrix_int(int **distanceMatrix){
  free_johnson_parallel_matrix<int>(distanceMatrix);
}

void johnson_parallel_matrix_successor_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int n){
  johnson_parallel_matrix<double>(adjacencyMatrix, distanceMatrix, successorMatrix, n);
}
void free_johnson_parallel_matrix_successor_double(double **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_matrix<double>(distanceMatrix, successorMatrix);
}
void johnson_parallel_matrix_successor_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int n){
  johnson_parallel_matrix<float>(adjacencyMatrix, distanceMatrix, successorMatrix, n);
}
void free_johnson_parallel_matrix_successor_float(float **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_matrix<float>(distanceMatrix, successorMatrix);
}
void johnson_parallel_matrix_successor_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int n){
  johnson_parallel_matrix<int>(adjacencyMatrix, distanceMatrix, successorMatrix, n);
}
void free_johnson_parallel_matrix_successor_int(int **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_matrix<int>(distanceMatrix, successorMatrix);
}

void johnson_parallel_list_double(const int v, const int e, const int* edges, const double* distances, double **distanceMatrix){
  johnson_parallel_list<double>(v, e, edges, distances, distanceMatrix);
}
void free_johnson_parallel_list_double(double **distanceMatrix){
  free_johnson_parallel_list<double>(distanceMatrix);
}
void johnson_parallel_list_float(const int v, const int e, const int* edges, const float* distances, float **distanceMatrix){
  johnson_parallel_list<float>(v, e, edges, distances, distanceMatrix);
}
void free_johnson_parallel_list_float(float **distanceMatrix){
  free_johnson_parallel_list<float>(distanceMatrix);
}
void johnson_parallel_list_int(const int v, const int e, const int* edges, const int* distances, int **distanceMatrix){
  johnson_parallel_list<int>(v, e, edges, distances, distanceMatrix);
}
void free_johnson_parallel_list_int(int **distanceMatrix){
  free_johnson_parallel_list<int>(distanceMatrix);
}

void johnson_parallel_list_successor_double(const int v, const int e, const int* edges, const double* distances, double **distanceMatrix, int **successorMatrix){
  johnson_parallel_list<double>(v, e, edges, distances, distanceMatrix, successorMatrix);
}
void free_johnson_parallel_list_successor_double(double **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_list<double>(distanceMatrix, successorMatrix);
}

void johnson_parallel_list_successor_float(const int v, const int e, const int* edges, const float* distances, float **distanceMatrix, int **successorMatrix){
  johnson_parallel_list<float>(v, e, edges, distances, distanceMatrix, successorMatrix);
}
void free_johnson_parallel_list_successor_float(float **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_list<float>(distanceMatrix, successorMatrix);
}

void johnson_parallel_list_successor_int(const int v, const int e, const int* edges, const int* distances, int **distanceMatrix, int **successorMatrix){
  johnson_parallel_list<int>(v, e, edges, distances, distanceMatrix, successorMatrix);
}
void free_johnson_parallel_list_successor_int(int **distanceMatrix, int **successorMatrix){
  free_johnson_parallel_list<int>(distanceMatrix, successorMatrix);
}
