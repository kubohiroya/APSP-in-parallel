#include "johnson_float.hpp"

#define THREADS_PER_BLOCK 32

__constant__ graph_cuda_t_float graph_const;

__forceinline__
__device__ int min_distance_float(float *dist, char *visited, int n) {
  float min = FLT_INF;
  int min_index = 0;
  for (int v = 0; v < n; v++) {
    if (!visited[v] && dist[v] <= min) {
      min = dist[v];
      min_index = v;
    }
  }
  return min_index;
}

__global__ void dijkstra_kernel_float(float *distanceMatrix, int *successorMatrix, char *visited_global) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  int V = graph_const.V;

  if (s >= V) return;

  int *starts = graph_const.starts;
  float *weights = graph_const.weights;
  edge_t_float *edge_array = graph_const.edge_array;

  float *dist = &distanceMatrix[s * V];
  char *visited = &visited_global[s * V];
  for (int i = 0; i < V; i++) {
    dist[i] = FLT_INF;
    visited[i] = 0;
  }
  dist[s] = 0;
  for (int count = 0; count < V - 1; count++) {
    int u = min_distance_float(dist, visited, V);
    int u_start = starts[u];
    int u_end = starts[u + 1];
    float dist_u = dist[u];
    visited[u] = 1;
    for (int v_i = u_start; v_i < u_end; v_i++) {
      int v = edge_array[v_i].v;
      if (!visited[v] && dist_u != FLT_INF && dist_u + weights[v_i] < dist[v])
        dist[v] = dist_u + weights[v_i];
      successorMatrix[count] = 0; // FIXME
    }
  }
}

__global__ void bellman_ford_kernel_float(float *dist) {
  int E = graph_const.E;
  int e = threadIdx.x + blockDim.x * blockIdx.x;

  if (e >= E) return;
  float *weights = graph_const.weights;
  edge_t_float *edges = graph_const.edge_array;
  int u = edges[e].u;
  int v = edges[e].v;
  float new_dist = weights[e] + dist[u];
  // Make ATOMIC
  if (dist[u] != FLT_INF && new_dist < dist[v])
    atomicExch(&dist[v], new_dist); // Needs to have conditional be atomic too
}

__host__ bool bellman_ford_cuda_float(graph_cuda_t_float *gr, float *dist, int s) {
  int V = gr->V;
  int E = gr->E;
  edge_t_float *edges = gr->edge_array;
  float *weights = gr->weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < V; i++) {
    dist[i] = FLT_INF;
  }
  dist[s] = 0;

  float *device_dist;
  cudaMalloc(&device_dist, sizeof(float) * V);
  cudaMemcpy(device_dist, dist, sizeof(float) * V, cudaMemcpyHostToDevice);

  int blocks = (E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  for (int i = 1; i <= V - 1; i++) {
    bellman_ford_kernel_float<<<blocks, THREADS_PER_BLOCK>>>(device_dist);
    cudaThreadSynchronize();
  }

  cudaMemcpy(dist, device_dist, sizeof(float) * V, cudaMemcpyDeviceToHost);
  bool no_neg_cycle = true;

  // use OMP to parallelize. Not worth sending to GPU
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < E; i++) {
    int u = edges[i].u;
    int v = edges[i].v;
    float weight = weights[i];
    if (dist[u] != FLT_INF && dist[u] + weight < dist[v])
      no_neg_cycle = false;
  }

  cudaFree(device_dist);

  return no_neg_cycle;
}

/**************************************************************************
                        Johnson's Algorithm CUDA
**************************************************************************/

__host__ void johnson_cuda_float(graph_cuda_t_float *gr, float *distanceMatrix, int *successorMatrix) {

  //cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);

    std::cout << "Device " << i << ": " << deviceProps.name << "\n"
              << "\tSMs: " << deviceProps.multiProcessorCount << "\n"
              << "\tGlobal mem: " << static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024)
              << "GB \n"
              << "\tCUDA Cap: " << deviceProps.major << "." << deviceProps.minor << "\n";
  }

  // Const Graph Initialization
  int V = gr->V;
  int E = gr->E;
  // Structure of the graph
  edge_t_float *device_edge_array;
  float *device_weights;
  float *device_distanceMatrix;
  int *device_successorMatrix;
  int *device_starts;
  // Needed to run dijkstra
  char *device_visited;
  // Allocating memory
  cudaMalloc(&device_edge_array, sizeof(edge_t_float) * E);
  cudaMalloc(&device_weights, sizeof(float) * E);
  cudaMalloc(&device_distanceMatrix, sizeof(float) * V * V);
  cudaMalloc(&device_successorMatrix, sizeof(int) * V * V);
  cudaMalloc(&device_visited, sizeof(char) * V * V);
  cudaMalloc(&device_starts, sizeof(int) * (V + 1));

  cudaMemcpy(device_edge_array, gr->edge_array, sizeof(edge_t_float) * E,
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_weights, gr->weights, sizeof(float) * E, cudaMemcpyHostToDevice);
  cudaMemcpy(device_starts, gr->starts, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);

  graph_cuda_t_float graph_params;
  graph_params.V = V;
  graph_params.E = E;
  graph_params.starts = device_starts;
  graph_params.weights = device_weights;
  graph_params.edge_array = device_edge_array;
  // Constant memory parameters
  cudaMemcpyToSymbol(graph_const, &graph_params, sizeof(graph_cuda_t_float));
  // End initialization

  graph_cuda_t_float *bf_graph = new graph_cuda_t_float;
  bf_graph->V = V + 1;
  bf_graph->E = gr->E + V;
  bf_graph->edge_array = new edge_t_float[bf_graph->E];
  bf_graph->weights = new float[bf_graph->E];

  std::memcpy(bf_graph->edge_array, gr->edge_array, gr->E * sizeof(edge_t_float));
  std::memcpy(bf_graph->weights, gr->weights, gr->E * sizeof(float));
  std::memset(&bf_graph->weights[gr->E], 0, V * sizeof(float));

  float *h = new float[bf_graph->V];
  bool r = bellman_ford_cuda_float(bf_graph, h, V);
  if (!r) {
    std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
    exit(1);
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int e = 0; e < E; e++) {
    int u = gr->edge_array[e].u;
    int v = gr->edge_array[e].v;
    gr->weights[e] = gr->weights[e] + h[u] - h[v];
  }

  int blocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  cudaMemcpy(device_weights, gr->weights, sizeof(float) * E, cudaMemcpyHostToDevice);

  dijkstra_kernel_float<<<blocks, THREADS_PER_BLOCK>>>(device_distanceMatrix, device_successorMatrix, device_visited);

  cudaMemcpy(distanceMatrix, device_distanceMatrix, sizeof(float) * V * V, cudaMemcpyDeviceToHost);

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
  cudaFree(device_successorMatrix);
  cudaFree(device_starts);
  cudaFree(device_visited);

}

