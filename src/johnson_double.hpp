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

graph_t_double *johnson_init_double(const int n, const double p, const unsigned long seed);

typedef struct edge_double {
  int u;
  int v;
} edge_t_double;

typedef struct graph_cuda_double {
  int V;
  int E;
  int *starts;
  double *weights;
  edge_t_double *edge_array;
} graph_cuda_t_double;


graph_cuda_t_double *johnson_cuda_init_double(const int n, const double p, const unsigned long seed);

void johnson_cuda_double(graph_cuda_t_double *gr, double *output, int *parents);

void free_cuda_graph_double(graph_cuda_t_double *g);

void free_graph_double(graph_t_double *g);

void johnson_parallel_double(graph_t_double *gr, double *output, int *parents);

