#pragma once

#include <iostream> // cerr
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

#include "inf.hpp"
#include "getInf.hpp"
#include "util.hpp"

using namespace boost;

template<typename Number> using Graph = adjacency_list<listS, vecS, directedS, no_property, property<edge_weight_t, Number>>;
template<typename Number> using Vertex = typename graph_traits<Graph<Number>>::vertex_descriptor;
typedef std::pair<int, int> Edge;

template<typename Number>
struct graph_t {
  int V;
  int E;
  Edge *edge_array;
  Number *weights;
};

#ifdef CUDA
template<typename Number>
struct graph_cuda_t {
  int V;
  int E;
  Edge *edge_array;
  Number *weights;
  int *starts;
};
#endif

template<typename Number> size_t init_random_adjacency_matrix(Number *adjacencyMatrix, const int n, const double p, const unsigned long seed) {
  static const Number inf = getInf<Number>();
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_real_distribution<double> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  size_t e = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adjacencyMatrix[i * n + j] = 0;
      } else if (flip(rand_engine) < p) {
        adjacencyMatrix[i * n + j] = choose_weight(rand_engine);
        e++;
      } else {
        adjacencyMatrix[i * n + j] = inf;
      }
    }
  }
  return e;
}

template<typename Number> size_t count_edges(const Number *adjacencyMatrix, const int n) {
  static const Number inf = getInf<Number>();
  size_t e = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n * n; i++) {
    Number weight = adjacencyMatrix[i];
    if (weight != 0 && weight != inf) {
#ifdef _OPENMP
#pragma omp atomic
#endif
      e++;
    }
  }
  return e;
}

template<typename Number> graph_t<Number> * init_graph(const Number *adjacencyMatrix, const int n) {
  static const Number inf = getInf<Number>();
  size_t e = count_edges<Number>(adjacencyMatrix, n);
  Edge *edge_array = new Edge[e];
  Number *weights = new Number[e];
  int ei = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (adjacencyMatrix[i * n + j] != 0 && adjacencyMatrix[i * n + j] != inf) {
#ifdef _OPENMP
#pragma omp critical (init_graph)
#endif
        {
          edge_array[ei] = Edge(i, j);
          weights[ei] = adjacencyMatrix[i * n + j];
          ei++;
        }
      }
    }
  }
  graph_t<Number> *gr = new graph_t<Number>;
  gr->V = n;
  gr->E = e;
  gr->edge_array = edge_array;
  gr->weights = weights;
  return gr;
}

template<typename Number> graph_t<Number> * init_random_graph(const int n, const double p, const unsigned long seed) {
  static const Number inf = getInf<Number>();
  Number *adjacencyMatrix = new Number[n * n];
  size_t e = init_random_adjacency_matrix<Number>(adjacencyMatrix, n, p, seed);
  graph_t<Number> * gr = init_graph<Number>(adjacencyMatrix, n, e);
  delete[] adjacencyMatrix;
  return gr;
}

template<typename Number> void free_graph(const graph_t<Number> *g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete g;
}

#ifdef CUDA

template<typename Number> graph_cuda_t<Number> * init_graph_cuda(const Number *adjacencyMatrix, const int n) {
  static const Number inf = getInf<Number>();
  size_t e = count_edges<Number>(adjacencyMatrix, n);
  Edge *edge_array = new Edge[e];
  Number *weights = new Number[e];
  int* starts = new int[n + 1];  // Starting point for each edge
  int ei = 0;
  for (int i = 0; i < n; i++) {
    starts[i] = ei;
    for (int j = 0; j < n; j++) {
      if (adjacencyMatrix[i * n + j] != 0 && adjacencyMatrix[i * n + j] != inf) {
#ifdef _OPENMP
#pragma omp critical (init_graph_cuda)
#endif
        {
          edge_array[ei] = Edge(i, j);
          weights[ei] = adjacencyMatrix[i * n + j];
          ei++;
        }
      }
    }
  }

  starts[n] = ei; // One extra

  graph_cuda_t<Number> *gr = new graph_cuda_t<Number>;
  gr->V = n;
  gr->E = e;
  gr->edge_array = edge_array;
  gr->weights = weights;
  gr->starts = starts;
  return gr;
}

template<typename Number> graph_cuda_t<Number> * johnson_cuda_random_init(const int n, const double p, const unsigned long seed) {
  static const Number inf = getInf<Number>();
  Number* adjacencyMatrix = new Number[n * n];
  int e = init_random_adjacency_matrix<Number>(adjacencyMatrix, n, p, seed);
  graph_cuda_t<Number> *gr = init_graph_cuda<Number>(adjacencyMatrix, n);
  delete[] adjacencyMatrix;
  return gr;
}

template<typename Number> void free_graph_cuda(const graph_cuda_t<Number> * g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete[] g->starts;
  delete g;
}

template<typename Number> void johnson_cuda(graph_cuda_t<Number> *gr, Number *distanceMatrix);
template<typename Number> void johnson_successor_cuda(graph_cuda_t<Number> *gr, Number *distanceMatrix, int *successorMatrix);

#endif

template<typename Number> inline bool bellman_ford(const graph_t<Number> *gr, Number *dist, int src) {
  static const Number inf = getInf<Number>();
  int v = gr->V;
  int e = gr->E;
  Edge *edges = gr->edge_array;
  Number *weights = gr->weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < v; i++) {
    dist[i] = inf;
  }
  dist[src] = 0;

  for (int i = 1; i <= v - 1; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < e; j++) {
      int u = std::get<0>(edges[j]);
      int v = std::get<1>(edges[j]);
      Number new_dist = weights[j] + dist[u];
      if (dist[u] != inf && new_dist < dist[v])
        dist[v] = new_dist;
    }
  }

  bool no_neg_cycle = true;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < e; i++) {
    int u = std::get<0>(edges[i]);
    int v = std::get<1>(edges[i]);
    Number weight = weights[i];
    if (dist[u] != inf && dist[u] + weight < dist[v])
      no_neg_cycle = false;
  }
  return no_neg_cycle;
}

template<typename Number> void johnson_parallel(const graph_t<Number> *gr, Number *distanceMatrix) {
#ifdef _OPENMP
  if(thread_count > 1){
    omp_set_num_threads(thread_count);
  }
#endif

  static const Number inf = getInf<Number>();
  int v = gr->V;

  // Make new graph for Bellman-Ford
  // First, a new node q is added to the graph, connected by zero-weight edges
  // to each of the other nodes.
  graph_t<Number> *bf_graph = new graph_t<Number>;
  bf_graph->V = v + 1;
  bf_graph->E = gr->E + v;
  bf_graph->edge_array = new Edge[bf_graph->E];
  bf_graph->weights = new Number[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, gr->E * sizeof(Edge));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(Number));
  std::memset(&bf_graph->weights[gr->E], 0, v * sizeof(Number));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < v; e++) {
    bf_graph->edge_array[e + gr->E] = Edge(v, e);
  }

  // Second, the Bellman–Ford algorithm is used, starting from the new vertex q,
  // to find for each vertex v the minimum weight h(v) of a path from q to v. If
  // this step detects a negative cycle, the algorithm is terminated.
  // TODO Can run parallel version?
  Number *h = new Number[bf_graph->V];
  bool r = bellman_ford<Number>(bf_graph, h, v);
  if (!r) {
    std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
    exit(1);
  }

  // Next the edges of the original graph are reweighted using the values computed
  // by the Bellman–Ford algorithm: an edge from u to v, having length
  // w(u,v), is given the new length w(u,v) + h(u) − h(v).
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < gr->E; e++) {
    int u = std::get<0>(gr->edge_array[e]);
    int v = std::get<1>(gr->edge_array[e]);
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  Graph<Number> G(gr->edge_array, gr->edge_array + gr->E, gr->weights, v);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int s = 0; s < v; s++) {
    std::vector <Vertex<Number>> p(num_vertices(G));
    std::vector<Number> d(num_vertices(G));
    dijkstra_shortest_paths(G, s, distance_map(&d[0]).distance_inf(inf));
    for (int vi = 0; vi < v; vi++) {
      int i = s * v + vi;
      distanceMatrix[i] = d[vi] + h[vi] - h[s];
    }
  }

  delete[] h;
  free_graph<Number>(bf_graph);
}

template<typename Number> void johnson_parallel(const graph_t<Number> *gr, Number *distanceMatrix, int *successorMatrix) {
#ifdef _OPENMP
  if(thread_count > 1){
    omp_set_num_threads(thread_count);
  }
#endif


  static const Number inf = getInf<Number>();
  int v = gr->V;

  // Make new graph for Bellman-Ford
  // First, a new node q is added to the graph, connected by zero-weight edges
  // to each of the other nodes.
  graph_t<Number> *bf_graph = new graph_t<Number>;
  bf_graph->V = v + 1;
  bf_graph->E = gr->E + v;
  bf_graph->edge_array = new Edge[bf_graph->E];
  bf_graph->weights = new Number[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, gr->E * sizeof(Edge));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(Number));
  std::memset(&bf_graph->weights[gr->E], 0, v * sizeof(Number));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < v; e++) {
    bf_graph->edge_array[e + gr->E] = Edge(v, e);
  }

  // Second, the Bellman–Ford algorithm is used, starting from the new vertex q,
  // to find for each vertex v the minimum weight h(v) of a path from q to v. If
  // this step detects a negative cycle, the algorithm is terminated.
  // TODO Can run parallel version?
  Number *h = new Number[bf_graph->V];
  bool r = bellman_ford<Number>(bf_graph, h, v);
  if (!r) {
    std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
    exit(1);
  }

  // Next the edges of the original graph are reweighted using the values computed
  // by the Bellman–Ford algorithm: an edge from u to v, having length
  // w(u,v), is given the new length w(u,v) + h(u) − h(v).
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < gr->E; e++) {
    int u = std::get<0>(gr->edge_array[e]);
    int v = std::get<1>(gr->edge_array[e]);
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  Graph<Number> G(gr->edge_array, gr->edge_array + gr->E, gr->weights, v);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int s = 0; s < v; s++) {
    std::vector <Vertex<Number>> p(num_vertices(G));
    std::vector<Number> d(num_vertices(G));
    dijkstra_shortest_paths(G, s, distance_map(&d[0]).predecessor_map(&p[0]).distance_inf(inf));
    for (int vi = 0; vi < v; vi++) {
      int i = s * v + vi;
      distanceMatrix[i] = d[vi] + h[vi] - h[s];
      successorMatrix[vi * v + s] = p[vi];
    }
  }

  delete[] h;
  free_graph<Number>(bf_graph);
}

template<typename Number> void johnson_parallel_matrix(const Number *adjacencyMatrix, Number **distanceMatrix, const int n) {
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);
  memcpy(*distanceMatrix, adjacencyMatrix, sizeof(Number) * n * n);
#ifdef CUDA
  graph_cuda_t<Number> *cuda_gr = init_graph_cuda<Number>(adjacencyMatrix, n);
  johnson_cuda<Number>(cuda_gr, *distanceMatrix);
  free_graph_cuda<Number>(cuda_gr);
#else
  const graph_t<Number> *gr = init_graph<Number>(adjacencyMatrix, n);
  johnson_parallel<Number>(gr, *distanceMatrix);
  delete gr;
#endif
}

template<typename Number> void johnson_parallel_matrix(const Number *adjacencyMatrix, Number **distanceMatrix, int **successorMatrix, const int n) {
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);
  *successorMatrix = (int *) malloc(sizeof(int) * n * n);
  memcpy(*distanceMatrix, adjacencyMatrix, sizeof(Number) * n * n);
#ifdef CUDA
  graph_cuda_t<Number> *cuda_gr = init_graph_cuda<Number>(adjacencyMatrix, n);
  johnson_successor_cuda<Number>(cuda_gr, *distanceMatrix, *successorMatrix);
  free_graph_cuda<Number>(cuda_gr);
#else
    const graph_t<Number> *gr = init_graph<Number>(adjacencyMatrix, n);
  johnson_parallel<Number>(gr, *distanceMatrix, *successorMatrix);
  delete gr;
#endif
}

template<typename Number> void free_johnson_parallel_matrix(Number **distanceMatrix) {
  free(*distanceMatrix);
}
template<typename Number> void free_johnson_parallel_matrix(Number **distanceMatrix, int **successorMatrix) {
  free(*distanceMatrix);
  free(*successorMatrix);
}

extern "C" void johnson_parallel_matrix_double(const double *adjacencyMatrix, double **distanceMatrix, const int n);
extern "C" void free_johnson_parallel_matrix_double(double **distanceMatrix);
extern "C" void johnson_parallel_matrix_float(const float *adjacencyMatrix, float **distanceMatrix, const int n);
extern "C" void free_johnson_parallel_matrix_float(float **distanceMatrix);
extern "C" void johnson_parallel_matrix_int(const int *adjacencyMatrix, int **distanceMatrix, const int n);
extern "C" void free_johnson_parallel_matrix_int(int **distanceMatrix);

extern "C" void johnson_parallel_matrix_successor_double(const double *adjacencyMatrix, double **distanceMatrix, int **successorMatrix, const int n);
extern "C" void free_johnson_parallel_matrix_successor_double(double **distanceMatrix, int **successorMatrix);
extern "C" void johnson_parallel_matrix_successor_float(const float *adjacencyMatrix, float **distanceMatrix, int **successorMatrix, const int n);
extern "C" void free_johnson_parallel_matrix_successor_float(float **distanceMatrix, int **successorMatrix);
extern "C" void johnson_parallel_matrix_successor_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int n);
extern "C" void free_johnson_parallel_matrix_successor_int(int **distanceMatrix, int **successorMatrix);
