#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "inf.hpp"

using namespace boost;

typedef adjacency_list <listS, vecS, directedS,
no_property, property<edge_weight_t, int>> Graph_int;
typedef graph_traits<Graph_int>::vertex_descriptor Vertex_int;
typedef std::pair<int, int> Edge_int;

typedef struct graph_int {
  int V;
  int E;
  Edge_int *edge_array;
  int *weights;
} graph_t_int;

typedef struct edge_int {
  int u;
  int v;
} edge_t_int;

#ifdef CUDA
typedef struct graph_cuda_int {
  int V;
  int E;
  int *starts;
  int *weights;
  edge_t_int *edge_array;
} graph_cuda_t_int;
#endif

int init_random_adjacency_matrix_int(int *adjacencyMatrix, const int n, const double p, const unsigned long seed);

int count_edges_int(const int *adjacencyMatrix, const int n);

graph_t_int *init_random_graph_int(const int n, const double p, const unsigned long seed);

graph_t_int *init_graph_int(const int *adjacencyMatrix, const int n, const int e);

#ifdef CUDA
graph_cuda_t_int *johnson_cuda_random_init_int(const int n, const double p, const unsigned long seed);
graph_cuda_t_int *init_graph_cuda_int(const int *adjacencyMatrix, const int n, const int e);
void johnson_cuda_int(graph_cuda_t_int *gr, int *distanceMatrix, int *successorMatrix);
void free_cuda_graph_int(graph_cuda_t_int *g);
#endif

void free_graph_int(graph_t_int *g);

void johnson_parallel_int(graph_t_int *gr, int *distanceMatrix, int *successorMatrix);

extern "C" void johnson_parallel_matrix_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int n);
extern "C" void free_johnson_parallel_matrix_int(int **distanceMatrix, int **successorMatrix);

