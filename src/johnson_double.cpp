#include <iostream> // cerr
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include "johnson_double.hpp"

static double dbl_max = std::numeric_limits<double>::max();
graph_t_double *johnson_init_double(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_int_distribution<double> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  double *adj_matrix = new double[n * n];
  
  size_t E = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adj_matrix[i*n + j] = 0;
      } else if (flip(rand_engine) < p) {
        adj_matrix[i*n + j] = choose_weight(rand_engine);
        E ++;
      } else {
        adj_matrix[i*n + j] = dbl_max;
      }
    }
  }
  Edge_double *edge_array = new Edge_double[E];
  double *weights = new double[E];
  int ei = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (adj_matrix[i*n + j] != 0.0f
          && adj_matrix[i*n + j] != dbl_max) {
        edge_array[ei] = Edge_double(i,j);
        weights[ei] = adj_matrix[i*n + j];
        ei++;
      }
    }
  }

  delete[] adj_matrix;

  graph_t_double *gr = new graph_t_double;
  gr->V = n;
  gr->E = E;
  gr->edge_array = edge_array;
  gr->weights = weights;

  return gr;
}

#ifdef CUDA
void free_graph_cuda_double(graph_cuda_t_double *g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete g;
}

void set_edge_double(edge_t_double *edge, int u, int v) {
  edge->u = u;
  edge->v = v;
}

graph_cuda_t_double *johnson_cuda_init_double(const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_int_distribution<double> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  double *adj_matrix = new double[n * n];
  size_t E = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adj_matrix[i*n + j] = 0.0f;
      } else if (flip(rand_engine) < p) {
        adj_matrix[i*n + j] = choose_weight(rand_engine);
        E ++;
      } else {
        adj_matrix[i*n + j] = dbl_max;
      }
    }
  }
  edge_t_double *edge_array = new edge_t_double[E];
  int* starts = new int[n + 1];  // Starting point for each edge
  double* weights = new double[E];
  int ei = 0;
  for (int i = 0; i < n; i++) {
    starts[i] = ei;
    for (int j = 0; j < n; j++) {
      if (adj_matrix[i*n + j] != 0.0f
          && adj_matrix[i*n + j] != dbl_max) {
        set_edge(&edge_array[ei], i, j);
        weights[ei] = adj_matrix[i*n + j];
        ei++;
      }
    }
  }
  starts[n] = ei; // One extra

  delete[] adj_matrix;

  graph_cuda_t_double *gr = new graph_cuda_t_double;
  gr->V = n;
  gr->E = E;
  gr->edge_array = edge_array;
  gr->weights = weights;
  gr->starts = starts;

  return gr;
}

void free_cuda_graph_double(graph_cuda_t_double* g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete[] g->starts;
  delete g;
}

#endif

void free_graph_double(graph_t_double* g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete g;
}

inline bool bellman_ford_double(graph_t_double* gr, double* dist, int src) {
  int V = gr->V;
  int E = gr->E;
  Edge_double* edges = gr->edge_array;
  double* weights = gr->weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < V; i++) {
    dist[i] = dbl_max;
  }
  dist[src] = 0;


  for (int i = 1; i <= V-1; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < E; j++) {
      int u = std::get<0>(edges[j]);
      int v = std::get<1>(edges[j]);
      double new_dist = weights[j] + dist[u];
      if (dist[u] != dbl_max && new_dist < dist[v])
        dist[v] = new_dist;
    }
  }

  bool no_neg_cycle = true;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < E; i++) {
    int u = std::get<0>(edges[i]);
    int v = std::get<1>(edges[i]);
    double weight = weights[i];
    if (dist[u] != dbl_max && dist[u] + weight < dist[v])
      no_neg_cycle = false;
  }
  return no_neg_cycle;
}

void johnson_parallel_double(graph_t_double* gr, double* output) {

  int V = gr->V;

  // Make new graph for Bellman-Ford
  // First, a new node q is added to the graph, connected by zero-weight edges
  // to each of the other nodes.
  graph_t_double* bf_graph = new graph_t_double;
  bf_graph->V = V + 1;
  bf_graph->E = gr->E + V;
  bf_graph->edge_array = new Edge_double[bf_graph->E];
  bf_graph->weights = new double[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, gr->E  * sizeof(Edge_double));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(double));
  std::memset(&bf_graph->weights[gr->E], 0, V * sizeof(double));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < V; e++) {
    bf_graph->edge_array[e + gr->E] = Edge_double(V, e);
  }

  // Second, the Bellman–Ford algorithm is used, starting from the new vertex q,
  // to find for each vertex v the minimum weight h(v) of a path from q to v. If
  // this step detects a negative cycle, the algorithm is terminated.
  // TODO Can run parallel version?
  double* h = new double[bf_graph->V];
  bool r = bellman_ford_double(bf_graph, h, V);
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

  Graph_double G(gr->edge_array, gr->edge_array + gr->E, gr->weights, V);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int s = 0; s < V; s++) {
    std::vector<double> d(num_vertices(G));
    dijkstra_shortest_paths(G, s, distance_map(&d[0]));
    for (int v = 0; v < V; v++) {
      output[s*V + v] = d[v] + h[v] - h[s];
    }
  }

  delete[] h;
  free_graph_double(bf_graph);
}
