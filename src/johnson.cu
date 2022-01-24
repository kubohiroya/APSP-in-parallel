#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "johnson.hpp"

#define THREADS_PER_BLOCK 32

template<typename Number>
__forceinline__
__device__
int min_distance(Number *dist, char *visited, int n, Number inf) {
  Number min = inf;
  int min_index = 0;
  for (int v = 0; v < n; v++) {
    if (!visited[v] && dist[v] <= min) {
      min = dist[v];
      min_index = v;
    }
  }
  return min_index;
}

template<typename Number>
__global__
void dijkstra_kernel(graph_cuda_t<Number> *graph_const, Number *distanceMatrix, int *successorMatrix, char *visited_global, const Number inf) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  int V = graph_const->V;
  if (s >= graph_const->V) return;

  Number *dist = &distanceMatrix[s * V];
  char *visited = &visited_global[s * V];
  int *starts = graph_const->starts;
  Number *weights = graph_const->weights;
  Edge* edge_array = graph_const->edge_array;

  for (int i = 0; i < V; i++) {
    dist[i] = inf;
    visited[i] = 0;
  }
  dist[s] = 0;
  for (int count = 0; count < V - 1; count++) {
    int u = min_distance<Number>(dist, visited, V, inf);
    int u_start = starts[u];
    int u_end = starts[u + 1];
    Number dist_u = dist[u];
    visited[u] = 1;
    for (int v_i = u_start; v_i < u_end; v_i++) {
      int v = std::get<1>(edge_array[v_i]);
      if (!visited[v] && dist_u != inf && dist_u + weights[v_i] < dist[v]) {
        dist[v] = dist_u + weights[v_i];
        int u = std::get<0>(edge_array[v_i]);
	successorMatrix[v_i * v + s] = u;
      }
    }
  }
}


template<typename Number>
__device__ void _atomicAdd(Number *oldvalue, Number newValue);

template<> __device__ void _atomicAdd<int>(int *oldValue, int newValue){
  atomicAdd(oldValue, newValue);
}
template<> __device__ void _atomicAdd<float>(float *oldValue, float newValue){
  atomicAdd(oldValue, newValue);
}
template<> __device__ void _atomicAdd<double>(double *oldValue, double newValue){
  atomicAdd((unsigned long long int*) oldValue, __double_as_longlong(newValue));
}

template<typename Number>
__device__ void _atomicExch(Number *oldvalue, Number newValue);

template<> __device__ void _atomicExch<int>(int *oldValue, int newValue){
  atomicExch(oldValue, newValue);
}
template<> __device__ void _atomicExch<float>(float *oldValue, float newValue){
  atomicExch(oldValue, newValue);
}
template<> __device__ void _atomicExch<double>(double *oldValue, double newValue){
  atomicExch((unsigned long long int*) oldValue, __double_as_longlong(newValue));
}

template<typename Number>
__global__
void bellman_ford_kernel(graph_cuda_t<Number> *graph_const, Number *dist, const Number inf) {
  int E = graph_const->E;
  int e = threadIdx.x + blockDim.x * blockIdx.x;

  if (e >= E) return;
  Number *weights = graph_const->weights;
  Edge *edges = graph_const->edge_array;
  int u = std::get<0>(edges[e]);
  int v = std::get<1>(edges[e]);
  Number new_dist = weights[e];
  _atomicAdd<Number>(&new_dist, dist[u]);
  // Make ATOMIC
  if (dist[u] != inf && new_dist < dist[v])
    _atomicExch<Number>(&dist[v], new_dist); // Needs to have conditional be atomic too
}

template<typename Number>
__host__
bool bellman_ford_cuda(graph_cuda_t<Number> *graph_const, graph_cuda_t<Number> *gr, Number *dist, int s) {
  int V = gr->V;
  int E = gr->E;
  // Edge *edges = gr->edge_array;
  Number *weights = gr->weights;
  static const Number inf = getInf<Number>();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < V; i++) {
    dist[i] = inf;
  }
  dist[s] = 0;

  Number *device_dist;
  cudaMalloc(&device_dist, sizeof(Number) * V);
  cudaMemcpy(device_dist, dist, sizeof(Number) * V, cudaMemcpyHostToDevice);

  Edge *device_edges;
  cudaMalloc(&device_edges, sizeof(Edge) * V);
  cudaMemcpy(device_edges, gr->edge_array, sizeof(Edge) * V, cudaMemcpyHostToDevice);

  int blocks = (E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  for (int i = 1; i <= V - 1; i++) {
    bellman_ford_kernel<Number> <<<blocks, THREADS_PER_BLOCK>>>(graph_const, device_dist, inf);
    cudaThreadSynchronize();
  }

  cudaMemcpy(dist, device_dist, sizeof(Number) * V, cudaMemcpyDeviceToHost);
  bool no_neg_cycle = true;

 // use OMP to parallelize. Not worth sending to GPU
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < E; i++) {
    int u = std::get<0>(gr->edge_array[i]);
    int v = std::get<1>(gr->edge_array[i]);
    Number weight = weights[i];
    if (dist[u] != inf && dist[u] + weight < dist[v]){
      no_neg_cycle = false;
    }
  }

  cudaFree(device_dist);
  cudaFree(device_edges);

  return no_neg_cycle;
}

/**************************************************************************
                        Johnson's Algorithm CUDA
**************************************************************************/

template<typename Number>
__host__
void johnson_successor_cuda(graph_cuda_t<Number> *gr, Number *distanceMatrix, int *successorMatrix) {
  //cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  static Number inf = getInf<Number>();

  // Const Graph Initialization
  int V = gr->V;
  int E = gr->E;
  // Structure of the graph
  Edge *device_edge_array;
  Number *device_weights;
  Number *device_distanceMatrix;
  int *device_successorMatrix = nullptr;
  int *device_starts;
  // Needed to run dijkstra
  char *device_visited;

  // Allocating memory
  cudaMalloc(&device_edge_array, sizeof(Edge) * E);
  cudaMalloc(&device_weights, sizeof(Number) * E);
  cudaMalloc(&device_distanceMatrix, sizeof(Number) * V * V);
  if(successorMatrix != nullptr){
    cudaMalloc(&device_successorMatrix, sizeof(int) * V * V);
  }
  cudaMalloc(&device_visited, sizeof(char) * V * V);
  cudaMalloc(&device_starts, sizeof(int) * (V + 1));

  cudaMemcpy(device_edge_array, gr->edge_array, sizeof(Edge) * E, cudaMemcpyHostToDevice);
  cudaMemcpy(device_weights, gr->weights, sizeof(Number) * E, cudaMemcpyHostToDevice);
  cudaMemcpy(device_starts, gr->starts, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);

  graph_cuda_t<Number> graph_params;
  graph_params.V = V;
  graph_params.E = E;
  graph_params.starts = device_starts;
  graph_params.weights = device_weights;
  graph_params.edge_array = device_edge_array;
  // Constant memory parameters

  graph_cuda_t<Number> graph_const;
  cudaMemcpy(&graph_const, &graph_params, sizeof(graph_cuda_t<Number>), cudaMemcpyHostToDevice);

  // Constant memory parameters
  graph_cuda_t<Number> *bf_graph = new graph_cuda_t<Number>;
  bf_graph->V = V + 1;
  bf_graph->E = gr->E + V;
  bf_graph->edge_array = new Edge[bf_graph->E];
  bf_graph->weights = new Number[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, sizeof(Edge) * gr->E);
  std::memcpy(bf_graph->weights, gr->weights, sizeof(Number) * gr->E);
  std::memset(&bf_graph->weights[gr->E], 0, V * sizeof(Number));

  Number *h = new Number[bf_graph->V];
  bool r = bellman_ford_cuda<Number>(&graph_const, bf_graph, h, V);
  if (!r) {
    std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
    exit(1);
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < E; e++) {
    int u = std::get<0>(gr->edge_array[e]);
    int v = std::get<1>(gr->edge_array[e]);
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  int blocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  cudaMemcpy(device_weights, gr->weights, sizeof(Number) * E, cudaMemcpyHostToDevice);
  if(successorMatrix != nullptr){
    cudaMemcpy(device_successorMatrix, successorMatrix, sizeof(int) * V * V, cudaMemcpyHostToDevice);
  }

  int *starts = gr->starts;
  Number *weights = gr->weights;
  Edge *edge_array = gr->edge_array;

  dijkstra_kernel<Number> <<<blocks, THREADS_PER_BLOCK>>>(gr, device_distanceMatrix, device_successorMatrix, device_visited, inf);

  cudaMemcpy(distanceMatrix, device_distanceMatrix, sizeof(Number) * V * V, cudaMemcpyDeviceToHost);
  if(successorMatrix != nullptr){
    cudaMemcpy(successorMatrix, device_successorMatrix, sizeof(int) * V * V, cudaMemcpyDeviceToHost);
  }

  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    std::cerr << "WARNING: A CUDA error occured: code=" << errCode << "," <<
              cudaGetErrorString(errCode) << "\n";
  }

  // Remember to reweight edges back -- for every s reweight every v
  // Could do in a kernel launch or with OMP

  cudaFree(device_edge_array);
  cudaFree(device_weights);
  cudaFree(device_distanceMatrix);
  if(successorMatrix != nullptr){
    cudaFree(device_successorMatrix);
  }
  cudaFree(device_visited);
  cudaFree(device_starts);
}

template<typename Number>
__host__
void johnson_cuda(graph_cuda_t<Number> *gr, Number *distanceMatrix) {
  int *successorMatrix = new int[gr->V * gr->V];
  johnson_successor_cuda<Number>(gr, distanceMatrix, successorMatrix);
  free(successorMatrix);
}


template __host__ void johnson_cuda<double>(graph_cuda_t<double> *gr, double *distanceMatrix);
template __host__ void johnson_cuda<float>(graph_cuda_t<float> *gr, float *distanceMatrix);
template __host__ void johnson_cuda<int>(graph_cuda_t<int> *gr, int *distanceMatrix);
template __host__ void johnson_successor_cuda<double>(graph_cuda_t<double> *gr, double *distanceMatrix, int *successorMatrix);
template __host__ void johnson_successor_cuda<float>(graph_cuda_t<float> *gr, float *distanceMatrix, int *successorMatrix);
template __host__ void johnson_successor_cuda<int>(graph_cuda_t<int> *gr, int *distanceMatrix, int *successorMatrix);

