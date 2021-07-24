#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

#include "inf.hpp"

using namespace boost;

typedef adjacency_list <listS, vecS, directedS,
no_property, property<edge_weight_t, float>> Graph_float;
typedef graph_traits<Graph_float>::vertex_descriptor Vertex_float;
typedef std::pair<int, int> Edge_float;

typedef struct graph_float {
  int V;
  int E;
  Edge_float *edge_array;
  float *weights;
} graph_t_float;

typedef struct edge_float {
  int u;
  int v;
} edge_t_float;

#ifdef CUDA
typedef struct graph_cuda_float {
  int V;
  int E;
  int *starts;
  float *weights;
  edge_t_float *edge_array;
} graph_cuda_t_float;
#endif

int init_random_adj_matrix_float(float *adj_matrix, const int n, const double p, const unsigned long seed);

int count_edges_float(const float *adj_matrix, const int n);

graph_t_float *init_random_graph_float(const int n, const double p, const unsigned long seed);

graph_t_float *init_graph_float(const int *adj_matrix, const int n, const int e);

#ifdef CUDA
graph_cuda_t_float *johnson_cuda_random_init_float(const int n, const double p, const unsigned long seed);
void johnson_cuda_float(graph_cuda_t_float *gr, float *output, int *parents);
void free_cuda_graph_float(graph_cuda_t_float *g);
#endif

void free_graph_float(graph_t_float *g);

void johnson_parallel_float(graph_t_float *gr, float *output, int *parents);

extern "C" void johnson_parallel_matrix_float(const float *adj_matrix, float **output, int **parents, const int n);
extern "C" void free_johnson_parallel_matrix_float(float *output, int *parents);
