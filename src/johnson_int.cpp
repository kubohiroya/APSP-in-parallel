#include <iostream> // cerr
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include "johnson_int.hpp"

int init_random_adjacency_matrix_int(int *adjacencyMatrix, const int n, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int E = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        adjacencyMatrix[i * n + j] = 0;
      } else if (flip(rand_engine) < p) {
        adjacencyMatrix[i * n + j] = choose_weight(rand_engine);
        E++;
      } else {
        adjacencyMatrix[i * n + j] = INT_INF;
      }
    }
  }
  return E;
}

int count_edges_int(const int *adjacencyMatrix, const int n) {
  size_t E = 0;
#ifdef _OPENMP
  // #pragma omp parallel for
#endif
  for (int i = 0; i < n * n; i++) {
    int weight = adjacencyMatrix[i];
    if (weight != 0 && weight != INT_INF) {
#ifdef _OPENMP
#pragma omp atomic
#endif
      E++;
    }
  }
  return E;
}

graph_t_int *init_graph_int(const int *adjacencyMatrix, const int n, const int E) {
  Edge_int *edge_array = new Edge_int[E];
  int *weights = new int[E];
  int ei = 0;
#ifdef _OPENMP
  // #pragma omp parallel for
#endif
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (adjacencyMatrix[i * n + j] != 0
          && adjacencyMatrix[i * n + j] != INT_INF) {
#ifdef _OPENMP
#pragma omp critical (init_graph_int)
#endif	
        {
          edge_array[ei] = Edge_int(i, j);
          weights[ei] = adjacencyMatrix[i * n + j];
          ei++;
        }
      }
    }
  }
  graph_t_int *gr = new graph_t_int;
  gr->V = n;
  gr->E = E;
  gr->edge_array = edge_array;
  gr->weights = weights;
  return gr;
}

graph_t_int *init_random_graph_int(const int n, const double p, const unsigned long seed) {
  int *adjacencyMatrix = new int[n * n];
  size_t E = init_random_adjacency_matrix_int(adjacencyMatrix, n, p, seed);
  graph_t_int *gr = init_graph_int(adjacencyMatrix, n, E);
  delete[] adjacencyMatrix;
  return gr;
}

#ifdef CUDA
void free_graph_cuda_int(graph_cuda_t_int *g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete g;
}

void set_edge_int(edge_t_int *edge, int u, int v) {
  edge->u = u;
  edge->v = v;
}

graph_cuda_t_int *johnson_cuda_random_init_int(const int n, const double p, const unsigned long seed) {

  int *adjacencyMatrix = new int[n * n];
  int E = init_random_adjacency_matrix_int(adjacencyMatrix, n, p, seed);

  edge_t_int *edge_array = new edge_t_int[E];
  int* starts = new int[n + 1];  // Starting point for each edge
  int* weights = new int[E];
  int ei = 0;
  for (int i = 0; i < n; i++) {
    starts[i] = ei;
    for (int j = 0; j < n; j++) {
      if (adjacencyMatrix[i*n + j] != 0
          && adjacencyMatrix[i*n + j] != INT_INF) {
        set_edge_int(&edge_array[ei], i, j);
        weights[ei] = adjacencyMatrix[i*n + j];
        ei++;
      }
    }
  }
  starts[n] = ei; // One extra

  delete[] adjacencyMatrix;

  graph_cuda_t_int *gr = new graph_cuda_t_int;
  gr->V = n;
  gr->E = E;
  gr->edge_array = edge_array;
  gr->weights = weights;
  gr->starts = starts;

  return gr;
}

void free_cuda_graph_int(graph_cuda_t_int* g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete[] g->starts;
  delete g;
}

#endif

void free_graph_int(graph_t_int *g) {
  delete[] g->edge_array;
  delete[] g->weights;
  delete g;
}

inline bool bellman_ford_int(graph_t_int *gr, int *dist, int src) {
  int V = gr->V;
  int E = gr->E;
  Edge_int *edges = gr->edge_array;
  int *weights = gr->weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < V; i++) {
    dist[i] = INT_INF;
  }
  dist[src] = 0;


  for (int i = 1; i <= V - 1; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < E; j++) {
      int u = std::get<0>(edges[j]);
      int v = std::get<1>(edges[j]);
      int new_dist = weights[j] + dist[u];
      if (dist[u] != INT_INF && new_dist < dist[v])
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
    int weight = weights[i];
    if (dist[u] != INT_INF && dist[u] + weight < dist[v])
      no_neg_cycle = false;
  }
  return no_neg_cycle;
}

void johnson_parallel_int(graph_t_int *gr, int *distanceMatrix, int *successorMatrix) {

  int V = gr->V;

  // Make new graph for Bellman-Ford
  // First, a new node q is added to the graph, connected by zero-weight edges
  // to each of the other nodes.
  graph_t_int *bf_graph = new graph_t_int;
  bf_graph->V = V + 1;
  bf_graph->E = gr->E + V;
  bf_graph->edge_array = new Edge_int[bf_graph->E];
  bf_graph->weights = new int[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, gr->E * sizeof(Edge_int));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(int));
  std::memset(&bf_graph->weights[gr->E], 0, V * sizeof(int));

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < V; e++) {
    bf_graph->edge_array[e + gr->E] = Edge_int(V, e);
  }

  // Second, the Bellman–Ford algorithm is used, starting from the new vertex q,
  // to find for each vertex v the minimum weight h(v) of a path from q to v. If
  // this step detects a negative cycle, the algorithm is terminated.
  // TODO Can run parallel version?
  int *h = new int[bf_graph->V];
  bool r = bellman_ford_int(bf_graph, h, V);
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

  Graph_int G(gr->edge_array, gr->edge_array + gr->E, gr->weights, V);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int s = 0; s < V; s++) {
    std::vector <Vertex_int> p(num_vertices(G));
    std::vector<int> d(num_vertices(G));
    dijkstra_shortest_paths(G, s, distance_map(&d[0]).predecessor_map(&p[0]).distance_inf(INT_INF));
    for (int v = 0; v < V; v++) {
      int i = s * V + v;
      distanceMatrix[i] = d[v] + h[v] - h[s];
      successorMatrix[v*V+s] = p[v];
    }
  }

  delete[] h;
  free_graph_int(bf_graph);
}

void johnson_parallel_matrix_int(const int *adjacencyMatrix, int **distanceMatrix, int **successorMatrix, const int n) {
  *distanceMatrix = (int *) malloc(sizeof(int) * n * n);
  *successorMatrix = (int *) malloc(sizeof(int) * n * n);
  johnson_parallel_int(init_graph_int(adjacencyMatrix, n, count_edges_int(adjacencyMatrix, n)), *distanceMatrix, *successorMatrix);
}

void free_johnson_parallel_matrix_int(int **distanceMatrix, int **successorMatrix) {
  free(*distanceMatrix);
  free(*successorMatrix);
}
