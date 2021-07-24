#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

#include "inf.hpp"

using namespace boost;

typedef adjacency_list <listS, vecS, directedS,
no_property, property<edge_weight_t, double>> Graph_double;
typedef graph_traits<Graph_double>::vertex_descriptor Vertex_double;
typedef std::pair<int, int> Edge_double;

typedef struct graph_double {
  int V;
  int E;
  Edge_double *edge_array;
  double *weights;
} graph_t_double;

typedef struct edge_double {
  int u;
  int v;
} edge_t_double;

#ifdef CUDA
typedef struct graph_cuda_double {
  int V;
  int E;
  int *starts;
  double *weights;
  edge_t_double *edge_array;
} graph_cuda_t_double;
#endif

int init_random_adj_matrix_double(double *adj_matrix, const int n, const double p, const unsigned long seed);

int count_edges_double(const double *adj_matrix, const int n);

graph_t_double *init_random_graph_double(const int n, const double p, const unsigned long seed);

graph_t_double *init_graph_double(const int *adj_matrix, const int n, const int e);

#ifdef CUDA
graph_cuda_t_double *johnson_cuda_random_init_double(const int n, const double p, const unsigned long seed);
void johnson_cuda_double(graph_cuda_t_double *gr, double *output, int *parents);
void free_cuda_graph_double(graph_cuda_t_double *g);
#endif

void free_graph_double(graph_t_double *g);

void johnson_parallel_double(graph_t_double *gr, double *output, int *parents);

extern "C" void johnson_parallel_matrix_double(const double *adj_matrix, double **output, int **parents, const int n);
extern "C" void free_johnson_parallel_matrix_double(double *output, int *parents);
