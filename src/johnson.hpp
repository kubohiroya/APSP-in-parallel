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
  Edge *edges;
  Number *weights;
  int *starts;
};

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

template<typename Number> graph_t<Number> * init_graph_matrix(const Number *adjacencyMatrix, const int n) {
  static const Number inf = getInf<Number>();
  size_t e = count_edges<Number>(adjacencyMatrix, n);
  Edge *edges = new Edge[e];
  Number *weights = new Number[e];
  int ei = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (adjacencyMatrix[i * n + j] != 0 && adjacencyMatrix[i * n + j] != inf) {
#ifdef _OPENMP
#pragma omp critical (init_graph_matrix)
#endif
        {
          edges[ei] = Edge(i, j);
          weights[ei] = adjacencyMatrix[i * n + j];
          ei++;
        }
      }
    }
  }
  graph_t<Number> *gr = new graph_t<Number>;
  gr->V = n;
  gr->E = e;
  gr->edges = edges;
  gr->weights = weights;
  return gr;
}

template<typename Number> graph_t<Number> * init_graph_list(const int v, const int e, const int *edges, const Number *weights) {
  graph_t<Number> *gr = new graph_t<Number>;
  gr->V = v;
  gr->E = e;
  gr->edges = new Edge[e];
  gr->weights = new Number[e];
  for (int i = 0; i < e; i++) {
    gr->edges[i] = Edge(edges[i*2], edges[i*2+1]);
    gr->weights[i] = weights[i];
  }
  return gr;
}

template<typename Number> graph_t<Number> * init_random_graph(const int n, const double p, const unsigned long seed) {
  const Number inf = getInf<Number>();
  Number *adjacencyMatrix = new Number[n * n];
  size_t e = init_random_adjacency_matrix<Number>(adjacencyMatrix, n, p, seed);
  graph_t<Number> * gr = init_graph_matrix<Number>(adjacencyMatrix, n, e);
  delete[] adjacencyMatrix;
  return gr;
}

template<typename Number> void free_graph(const graph_t<Number> *g) {
  delete[] g->edges;
  delete[] g->weights;
  delete g;
}

#ifdef CUDA

template<typename Number> graph_t<Number> * init_graph_matrix_cuda(const Number *adjacencyMatrix, const int n) {
  const Number inf = getInf<Number>();
  size_t e = count_edges<Number>(adjacencyMatrix, n);
  int *edges = new int[e * 2];
    Number *weights = new Number[e];
  int* starts = new int[n + 1];  // Starting point for each edge
  int ei = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    starts[i] = ei;
    for (int j = 0; j < n; j++) {
      if (adjacencyMatrix[i * n + j] != 0 && adjacencyMatrix[i * n + j] != inf) {
#ifdef _OPENMP
#pragma omp critical (init_graph_matrix_cuda)
#endif
        {
          edges[ei * 2] = i;
          edges[ei * 2 + 1] = j;
          weights[ei] = adjacencyMatrix[i * n + j];
          ei++;
        }
      }
    }
  }

  starts[n] = ei; // One extra

  graph_t<Number> *gr = new graph_t<Number>;
  gr->V = n;
  gr->E = e;
  gr->edges = edges;
  gr->weights = weights;
  gr->starts = starts;
  return gr;
}

template<typename Number> graph_t<Number> * johnson_cuda_random_init(const int n, const double p, const unsigned long seed) {
  static const Number inf = getInf<Number>();
  Number* adjacencyMatrix = new Number[n * n];
  int e = init_random_adjacency_matrix<Number>(adjacencyMatrix, n, p, seed);
  graph_t<Number> *gr = init_graph_matrix_cuda<Number>(adjacencyMatrix, n);
  delete[] adjacencyMatrix;
  return gr;
}

template<typename Number> void free_graph_cuda(const graph_t<Number> * g) {
  delete[] g->edges;
  delete[] g->weights;
  delete[] g->starts;
  delete g;
}

template<typename Number> void johnson_cuda(graph_t<Number> *gr, Number *distanceMatrix);
template<typename Number> void johnson_successor_cuda(graph_t<Number> *gr, Number *distanceMatrix, int *successorMatrix);

#endif

template<typename Number> inline bool bellman_ford(const graph_t<Number> *gr, Number *dist, int src) {
  static const Number inf = getInf<Number>();
  int v = gr->V;
  int e = gr->E;
  Edge *edges = gr->edges;
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
      int u = edges[j].first;
      int v = edges[j].second;
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
    int u = edges[i].first;
    int v = edges[i].second;
    Number weight = weights[i];
    if (dist[u] != inf && dist[u] + weight < dist[v]) {
      no_neg_cycle = false;
    }
  }
  return no_neg_cycle;
}

template<typename Number> void johnson_parallel(const graph_t<Number> *gr, Number *distanceMatrix) {

  static const Number inf = getInf<Number>();
  int v = gr->V;

  // Make new graph for Bellman-Ford
  // First, a new node q is added to the graph, connected by zero-weight edges
  // to each of the other nodes.
  graph_t<Number> *bf_graph = new graph_t<Number>;
  bf_graph->V = v + 1;
  bf_graph->E = gr->E + v;
  bf_graph->edges = new Edge[bf_graph->E];
  bf_graph->weights = new Number[bf_graph->E];

  std::memcpy(bf_graph->edges, gr->edges, sizeof(Edge) * gr->E);
  std::memcpy(bf_graph->weights, gr->weights, sizeof(Number) * gr->E);
  std::memset(&bf_graph->weights[gr->E], 0, sizeof(Number) * v);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < v; e++) {
    bf_graph->edges[e + gr->E] = Edge(v, e);
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
    int u = std::get<0>(gr->edges[e]);
    int v = std::get<1>(gr->edges[e]);
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  Graph<Number> G(gr->edges, gr->edges + gr->E, gr->weights, v);

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

  static const Number inf = getInf<Number>();
  int v = gr->V;

  // Make new graph for Bellman-Ford
  // First, a new node q is added to the graph, connected by zero-weight edges
  // to each of the other nodes.
  graph_t<Number> *bf_graph = new graph_t<Number>;
  bf_graph->V = v + 1;
  bf_graph->E = gr->E + v;
  bf_graph->edges = new Edge[bf_graph->E];
  bf_graph->weights = new Number[bf_graph->E];

  std::memcpy(bf_graph->edges, gr->edges, gr->E * sizeof(Edge));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(Number));
  std::memset(&bf_graph->weights[gr->E], 0, v * sizeof(Number));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < v; e++) {
    bf_graph->edges[e + gr->E] = Edge(v, e);
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
    int u = std::get<0>(gr->edges[e]);
    int v = std::get<1>(gr->edges[e]);
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  Graph<Number> G(gr->edges, gr->edges + gr->E, gr->weights, v);

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
#ifdef CUDA
  graph_t<Number> *gr = init_graph_matrix_cuda<Number>(adjacencyMatrix, n);
  johnson_cuda<Number>(gr, *distanceMatrix);
  free_graph_cuda<Number>(gr);
  return;
#else
  graph_t<Number> *gr = init_graph_matrix<Number>(adjacencyMatrix, n);
  johnson_parallel<Number>(gr, *distanceMatrix);
  free_graph<Number>(gr);
  return;
#endif
}
template<typename Number> void johnson_parallel_matrix(const Number *adjacencyMatrix, Number **distanceMatrix, int **successorMatrix, const int n) {
  *distanceMatrix = (Number *) malloc(sizeof(Number) * n * n);
  *successorMatrix = (int *) malloc(sizeof(int) * n * n);
#ifdef CUDA
  graph_t<Number> *gr = init_graph_matrix_cuda<Number>(adjacencyMatrix, n);
  johnson_successor_cuda<Number>(gr, *distanceMatrix, *successorMatrix);
  free_graph_cuda<Number>(gr);
  return;
#else
  memset(*successorMatrix, 0, sizeof(int) * n * n);
  graph_t<Number> *gr = init_graph_matrix<Number>(adjacencyMatrix, n);
  johnson_parallel<Number>(gr, *distanceMatrix, *successorMatrix);
  free_graph<Number>(gr);
  return;
#endif
}

template<typename Number> void johnson_parallel_list(const int v, const int e, const int* edges, const Number* distances, Number **distanceMatrix) {
  *distanceMatrix = (Number *) malloc(sizeof(Number) * v * v);
  graph_t<Number> * gr = init_graph_list<Number>(v, e, edges, distances);
#ifdef CUDA
  johnson_cuda<Number>(gr, *distanceMatrix);
  free_graph_cuda<Number>(gr);
  return;
#else
  johnson_parallel<Number>(gr, *distanceMatrix);
  free_graph<Number>(gr);
  return;
#endif
}

template<typename Number> void johnson_parallel_list(const int v, const int e, const int* edges, const Number* distances, Number **distanceMatrix, int **successorMatrix) {
  graph_t<Number> * gr = init_graph_list<Number>(v, e, edges, distances);
  *distanceMatrix = (Number *) malloc(sizeof(Number) * v * v);
  *successorMatrix = (int *) malloc(sizeof(int) * v * v);
#ifdef CUDA
  johnson_successor_cuda<Number>(gr, *distanceMatrix, *successorMatrix);
  free_graph_cuda<Number>(gr);
  return;
#else
  memset(*successorMatrix, 0, sizeof(int) * v * v);
  johnson_parallel<Number>(gr, *distanceMatrix, *successorMatrix);
  free_graph<Number>(gr);
  return;
#endif
}

template<typename Number> void free_johnson_parallel_matrix(Number **distanceMatrix) {
  delete[] *distanceMatrix;
}
template<typename Number> void free_johnson_parallel_matrix(Number **distanceMatrix, int **successorMatrix) {
  delete[] *distanceMatrix;
  delete[] *successorMatrix;
}

template<typename Number> void free_johnson_parallel_list(Number **distanceMatrix) {
  delete[] *distanceMatrix;
}
template<typename Number> void free_johnson_parallel_list(Number **distanceMatrix, int **successorMatrix) {
  delete[] *distanceMatrix;
  delete[] *successorMatrix;
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

extern "C" void johnson_parallel_list_double(const int v, const int e, const int* edges, const double* distances, double **distanceMatrix);
extern "C" void free_johnson_parallel_list_double(double **distanceMatrix);
extern "C" void johnson_parallel_list_float(const int v, const int e, const int* edges, const float* distances, float **distanceMatrix);
extern "C" void free_johnson_parallel_list_float(float **distanceMatrix);
extern "C" void johnson_parallel_list_int(const int v, const int e, const int* edges, const int* distances, int **distanceMatrix);
extern "C" void free_johnson_parallel_list_int(int **distanceMatrix);

extern "C" void johnson_parallel_list_successor_double(const int v, const int e, const int* edges, const double* distances, double **distanceMatrix, int **successorMatrix);
extern "C" void free_johnson_parallel_list_successor_double(double **distanceMatrix, int **successorMatrix);
extern "C" void johnson_parallel_list_successor_float(const int v, const int e, const int* edges, const float* distances, float **distanceMatrix, int **successorMatrix);
extern "C" void free_johnson_parallel_list_successor_float(float **distanceMatrix, int **successorMatrix);
extern "C" void johnson_parallel_list_successor_int(const int v, const int e, const int* edges, const int* distances, int **distanceMatrix, int **successorMatrix);
extern "C" void free_johnson_parallel_list_successor_int(int **distanceMatrix, int **successorMatrix);
