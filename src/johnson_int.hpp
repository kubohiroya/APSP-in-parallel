#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "inf.hpp"

using namespace boost;

typedef adjacency_list <listS, vecS, directedS,
no_property, property<edge_weight_t, int>> Graph_int;
typedef graph_traits<Graph_int>::vertex_descriptor Vertex_int;
typedef std::pair<int, int> Edge_int;

typedef struct graph {
  int V;
  int E;
  Edge_int *edge_array;
  int *weights;
} graph_t_int;

typedef struct edge {
  int u;
  int v;
} edge_t_int;

#ifdef CUDA
typedef struct graph_cuda {
  int V;
  int E;
  int *starts;
  int *weights;
  edge_t_int *edge_array;
} graph_cuda_t_int;
#endif

int init_random_adj_matrix_int(int *adj_matrix, const int n, const double p, const unsigned long seed);

int count_edges_int(const int *adj_matrix, const int n);

graph_t_int *init_random_graph_int(const int n, const double p, const unsigned long seed);

graph_t_int *init_graph_int(const int *adj_matrix, const int n, const int e);

#ifdef CUDA
graph_cuda_t_int *johnson_cuda_random_init_int(const int n, const double p, const unsigned long seed);
void johnson_cuda_int(graph_cuda_t_int *gr, int *output, int *parents);
void free_cuda_graph_int(graph_cuda_t_int *g);
#endif

void free_graph_int(graph_t_int *g);

void johnson_parallel_int(graph_t_int *gr, int *output, int *parents);

extern "C" void johnson_parallel_matrix_int(const int *adj_matrix, int **output, int **parents, const int n);
extern "C" void free_johnson_parallel_matrix_int(int **output, int **parents);

