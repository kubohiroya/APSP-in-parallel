#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "johnson.hpp"
#include "util.hpp"

#define THREADS_PER_BLOCK 32

#define HANDLE_ERROR(error) { \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} \

template<typename Number>
__forceinline__
__device__
int min_distance(Number *dist, int *visited, int n, Number inf) {
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
void bellman_ford_kernel(int E, Edge *edges, Number *weights, Number *distances, const Number inf) {
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  if (e >= E) return;
  int u = std::get<0>(edges[e]);
  int v = std::get<1>(edges[e]);
  Number new_distances = weights[e] + distances[u];
  // Make ATOMIC
  if (distances[u] != inf && new_distances < distances[v]) {
    _atomicExch<Number>(&distances[v], new_distances);
  }
}

template<typename Number>
__global__
void reweighting_kernel(int E, Edge *edges, Number *weights, Number *distances) {
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  if (e >= E) return;
  int u = std::get<0>(edges[e]);
  int v = std::get<1>(edges[e]);
  // Make ATOMIC?
  _atomicExch<Number>(&weights[e], weights[e] + distances[u] - distances[v]);
}

template<typename Number>
__global__
void check_negative_cycle_kernel(int E, Edge *edges, Number *weights, Number *distances, Number inf, int *neg_cycle) {
  int e = threadIdx.x + blockDim.x * blockIdx.x;
  if (e >= E) return;
  int u = std::get<0>(edges[e]);
  int v = std::get<1>(edges[e]);
  // Make ATOMIC?
  if (distances[u] != inf && distances[u] + weights[e] < distances[v]){
    _atomicExch<int>(neg_cycle, 1);
  }
}

template<typename Number>
__global__
void dijkstra_kernel(int V, int *starts, Number *weights, Edge *edges, Number *distanceMatrix, int *successorMatrix, int *visited_global, const Number inf) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= V) return;

  Number *distance = &distanceMatrix[s * V];
  int *visited = &visited_global[s * V];

  for (int i = 0; i < V; i++) {
    distance[i] = inf;
    visited[i] = 0;
  }
  distance[s] = 0;

  for(int count = 0; count < V; count++){
    int u = min_distance<Number>(distance, visited, V, inf);
    int u_start = starts[u];
    int u_end = starts[u + 1];
    Number distance_u = distance[u];
    _atomicExch<int>(&visited[u], 1);
    for (int v_i = u_start; v_i < u_end; v_i++) {
      int v = std::get<1>(edges[v_i]);
      if (!visited[v] && distance_u != inf && weights[v_i] != inf && distance_u + weights[v_i] < distance[v]) {
	_atomicExch<Number>(&distance[v], distance_u + weights[v_i]);
	int u = std::get<0>(edges[v_i]);
	successorMatrix[v_i * v + s] = u;
      }
    }
  }

}


/*
template<typename Number>
__host__
bool bellman_ford_cuda(graph_cuda_t<Number> *gr, Number *dist, int src) {
  static const Number inf = getInf<Number>();
  Edge *edges = gr->edges;
  Number *weights = gr->weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < gr->V; i++) {
    dist[i] = inf;
  }
  dist[src] = 0;

  Number *device_dist;
  cudaMalloc(&device_dist, sizeof(Number) * gr->V);
  cudaMemcpy(device_dist, dist, sizeof(Number) * gr->V , cudaMemcpyHostToDevice);

  Number *device_weights;
  cudaMalloc(&device_weights, sizeof(Number) * gr->E);
  cudaMemcpy(device_weights, weights, sizeof(Number) * gr->E, cudaMemcpyHostToDevice);

  Edge *device_edges;
  cudaMalloc(&device_edges, sizeof(Edge) * gr->E);
  cudaMemcpy(device_edges, edges, sizeof(Edge) * gr->E, cudaMemcpyHostToDevice);

  int blocks = (gr->E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  for (int i = 1; i <= gr->V - 1; i++) {
    bellman_ford_kernel<Number> <<<blocks, THREADS_PER_BLOCK>>>(E, device_edges, device_weights, device_dist, inf);
    cudaThreadSynchronize();
  }

  cudaMemcpy(dist, device_dist, sizeof(Number) * gr->V , cudaMemcpyDeviceToHost);
  cudaMemcpy(weights, device_weights, sizeof(Number) * gr->E, cudaMemcpyDeviceToHost);
  cudaMemcpy(edges, device_edges, sizeof(Edge) * gr->E, cudaMemcpyDeviceToHost);

  cudaFree(device_dist);
  cudaFree(device_weights);
  cudaFree(device_edges);

  bool no_neg_cycle = true;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < E; i++) {
    int u = std::get<0>(edges[i]);
    int v = std::get<1>(edges[i]);
    Number weight = weights[i];
    if (dist[u] != inf && dist[u] + weight < dist[v]){
      no_neg_cycle = false;
    }
  }

  return no_neg_cycle;
}
*/

/**************************************************************************
                        Johnson's Algorithm CUDA
**************************************************************************/

template<typename Number>
__host__
void johnson_successor_cuda(graph_cuda_t<Number> *gr, Number *distanceMatrix, int *successorMatrix) {
  // cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  Number inf = getInf<Number>();
  Number *distances = new Number[gr->V + 1];
  Edge *edges = new Edge[gr->V + gr->E];


#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int v = 0; v < gr->V + 1; v++) {
    distances[v] = inf;
  }
  distances[gr->V] = 0;

  Number *device_distances;
  HANDLE_ERROR(cudaMalloc(&device_distances, sizeof(Number) * (gr->V + 1)));
  HANDLE_ERROR(cudaMemcpy(device_distances, distances, sizeof(Number) * (gr->V + 1), cudaMemcpyHostToDevice));
  delete[] distances;

  Edge *device_edges;
  HANDLE_ERROR(cudaMalloc(&device_edges, sizeof(Edge) * (gr->E + gr->V)));
  std::memcpy(edges, gr->edges, sizeof(Edge) * gr->E);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < gr->V; e++){
    edges[e + gr->E] = Edge(gr->V, e);
  }
  HANDLE_ERROR(cudaMemcpy(device_edges, edges, sizeof(Edge) * (gr->E + gr->V), cudaMemcpyHostToDevice));

  Number *device_weights;
  HANDLE_ERROR(cudaMalloc(&device_weights, sizeof(Number) * gr->E));
  HANDLE_ERROR(cudaMemcpy(device_weights, gr->weights, sizeof(Number) * gr->E, cudaMemcpyHostToDevice));

  int bf_blocks = (gr->E + gr->V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  for(int i=0; i <= gr->V; i++){
    bellman_ford_kernel<Number> <<<bf_blocks, THREADS_PER_BLOCK>>>(gr->E + gr->V + 1, device_edges, device_weights, device_distances, inf);
    cudaThreadSynchronize();
  }

  int neg_cycle = 0;
  int *device_neg_cycle;
  HANDLE_ERROR(cudaMalloc(&device_neg_cycle, sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(device_neg_cycle, &neg_cycle, sizeof(int), cudaMemcpyHostToDevice));

  check_negative_cycle_kernel<Number> <<<bf_blocks, THREADS_PER_BLOCK>>>(gr->E + gr->V, device_edges, device_weights, device_distances, inf, device_neg_cycle);

  HANDLE_ERROR(cudaMemcpy(&neg_cycle, device_neg_cycle, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(device_neg_cycle);
  if (neg_cycle != 0) {
    std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
    exit(1);
  }

  reweighting_kernel<Number> <<<bf_blocks, THREADS_PER_BLOCK>>>(gr->E + gr->V, device_edges, device_weights, device_distances);

  int *device_starts;
  HANDLE_ERROR(cudaMalloc(&device_starts, sizeof(int) * (gr->V + 1)));
  HANDLE_ERROR(cudaMemcpy(device_starts, gr->starts, sizeof(int) * (gr->V + 1), cudaMemcpyHostToDevice));

  int *device_visited;
  HANDLE_ERROR(cudaMalloc(&device_visited, sizeof(int) * gr->V * gr->V));

  Number *device_distanceMatrix;
  HANDLE_ERROR(cudaMalloc(&device_distanceMatrix, sizeof(Number) * gr->V * gr->V));

  int *device_successorMatrix;
  HANDLE_ERROR(cudaMalloc(&device_successorMatrix, sizeof(int) * gr->V * gr->V));

  int dij_blocks = (gr->V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  dijkstra_kernel<Number> <<<dij_blocks, THREADS_PER_BLOCK>>>(gr->V, device_starts, device_weights, device_edges, device_distanceMatrix, device_successorMatrix, device_visited, inf);

  HANDLE_ERROR(cudaMemcpy(distanceMatrix, device_distanceMatrix, sizeof(Number) * gr->V * gr->V, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(successorMatrix, device_successorMatrix, sizeof(int) * gr->V * gr->V, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(device_distances));
  HANDLE_ERROR(cudaFree(device_edges));
  HANDLE_ERROR(cudaFree(device_weights));
  HANDLE_ERROR(cudaFree(device_starts));
  HANDLE_ERROR(cudaFree(device_visited));
  HANDLE_ERROR(cudaFree(device_distanceMatrix));
  HANDLE_ERROR(cudaFree(device_successorMatrix));
}

template<typename Number>
__host__
void johnson_cuda(graph_cuda_t<Number> *gr, Number *distanceMatrix) {
  int *successorMatrix = new int[gr->V * gr->V];
  johnson_successor_cuda<Number>(gr, distanceMatrix, successorMatrix);
  delete[] successorMatrix;
}

template __host__ void johnson_cuda<double>(graph_cuda_t<double> *gr, double *distanceMatrix);
template __host__ void johnson_cuda<float>(graph_cuda_t<float> *gr, float *distanceMatrix);
template __host__ void johnson_cuda<int>(graph_cuda_t<int> *gr, int *distanceMatrix);
template __host__ void johnson_successor_cuda<double>(graph_cuda_t<double> *gr, double *distanceMatrix, int *successorMatrix);
template __host__ void johnson_successor_cuda<float>(graph_cuda_t<float> *gr, float *distanceMatrix, int *successorMatrix);
template __host__ void johnson_successor_cuda<int>(graph_cuda_t<int> *gr, int *distanceMatrix, int *successorMatrix);

