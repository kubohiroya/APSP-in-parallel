#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "floyd_warshall.hpp"

#define BLOCK_DIM 16

__forceinline__
__host__ void check_cuda_error() {
  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    std::cerr << "WARNING: A CUDA error occured: code=" << errCode << "," <<
              cudaGetErrorString(errCode) << "\n";
  }
}

template<typename Number>
__forceinline__
__device__ void calc(Number *graph, int n, int k, int i, int j) {
  if ((i >= n) || (j >= n) || (k >= n)) return;
  const unsigned int kj = k * n + j;
  const unsigned int ij = i * n + j;
  const unsigned int ik = i * n + k;
  Number t1 = graph[ik] + graph[kj];
  Number t2 = graph[ij];
  graph[ij] = (t1 < t2) ? t1 : t2;
}


template<typename Number>
__global__ void floyd_warshall_kernel(int n, int k, Number *graph) {
  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  calc<Number>(graph, n, k, i, j);
}

/*****************************************************************************
                         Blocked Floyd-Warshall Kernel
  ***************************************************************************/

template<typename Number>
__forceinline__
__device__ void block_calc(Number *C, Number *A, Number *B, int bj, int bi) {
  for (int k = 0; k < BLOCK_DIM; k++) {
    Number sum = A[bi * BLOCK_DIM + k] + B[k * BLOCK_DIM + bj];
    if (C[bi * BLOCK_DIM + bj] > sum) {
      C[bi * BLOCK_DIM + bj] = sum;
    }
    __syncthreads();
  }
}

template<typename Number>
__global__ void floyd_warshall_block_kernel_phase1(int n, int k, Number *graph) {
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  __shared__ Number C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  // Transfer to temp shared arrays
  C[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];

  __syncthreads();

  block_calc<Number>(C, C, C, bi, bj);

  __syncthreads();

  // Transfer back to graph
  graph[k * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj] = C[bi * BLOCK_DIM + bj];

}

template<typename Number>
__global__ void floyd_warshall_block_kernel_phase2(int n, int k, Number *graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int i = blockIdx.x;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k) return;

  __shared__ Number A[BLOCK_DIM * BLOCK_DIM];
  __shared__ Number B[BLOCK_DIM * BLOCK_DIM];
  __shared__ Number C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi * BLOCK_DIM + bj] = graph[i * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];
  B[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];

  __syncthreads();

  block_calc<Number>(C, C, B, bi, bj);

  __syncthreads();

  graph[i * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj] = C[bi * BLOCK_DIM + bj];

  // Phase 2 1/2

  C[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + i * BLOCK_DIM + bi * n + bj];
  A[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];

  __syncthreads();

  block_calc<Number>(C, A, C, bi, bj);

  __syncthreads();

  // Block C is the only one that could be changed
  graph[k * BLOCK_DIM * n + i * BLOCK_DIM + bi * n + bj] = C[bi * BLOCK_DIM + bj];
}

template<typename Number>
__global__ void floyd_warshall_block_kernel_phase3(int n, int k, Number *graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int j = blockIdx.x;
  const unsigned int i = blockIdx.y;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k && j == k) return;
  __shared__ Number A[BLOCK_DIM * BLOCK_DIM];
  __shared__ Number B[BLOCK_DIM * BLOCK_DIM];
  __shared__ Number C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi * BLOCK_DIM + bj] = graph[i * BLOCK_DIM * n + j * BLOCK_DIM + bi * n + bj];
  A[bi * BLOCK_DIM + bj] = graph[i * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];
  B[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + j * BLOCK_DIM + bi * n + bj];

  __syncthreads();

  block_calc<Number>(C, A, B, bi, bj);

  __syncthreads();

  graph[i * BLOCK_DIM * n + j * BLOCK_DIM + bi * n + bj] = C[bi * BLOCK_DIM + bj];
}

/************************************************************************
                    Floyd-Warshall's Algorithm CUDA
************************************************************************/
template<typename Number>
__host__ void floyd_warshall_blocked_cuda(const Number *adjancencyMatrix, Number **distanceMatrix, int n) {

  Number *device_graph;
  const size_t size = sizeof(Number) * n * n;
  cudaMalloc(&device_graph, size);
  cudaMemcpy(device_graph, adjancencyMatrix, size, cudaMemcpyHostToDevice);

  //std::cout<<"floyd_warshall_blocked_cuda(const Number *adjancencyMatrix, Number **distanceMatrix, int n)\n";
  //print_matrix<Number>(adjancencyMatrix, n, n);

  const int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 phase4_grid(blocks, blocks, 1);

  for (int k = 0; k < blocks; k++) {
    floyd_warshall_block_kernel_phase1<Number> <<<1, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase2<Number> <<<blocks, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase3<Number> <<<phase4_grid, block_dim>>>(n, k, device_graph);
  }

  cudaMemcpy(*distanceMatrix, device_graph, size, cudaMemcpyDeviceToHost);
  check_cuda_error();
  cudaFree(device_graph);
}

template<typename Number>
__host__ void floyd_warshall_blocked_cuda(const Number *adjancencyMatrix, Number **distanceMatrix, int **successorMatrix, int n) {

  Number *device_graph;
  const size_t size = sizeof(Number) * n * n;
  cudaMalloc(&device_graph, size);
  cudaMemcpy(device_graph, adjancencyMatrix, size, cudaMemcpyHostToDevice);

  const int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 phase4_grid(blocks, blocks, 1);

  for (int k = 0; k < blocks; k++) {
    floyd_warshall_block_kernel_phase1<Number> <<<1, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase2<Number> <<<blocks, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase3<Number> <<<phase4_grid, block_dim>>>(n, k, device_graph);
  }


  cudaMemcpy(*distanceMatrix, device_graph, size, cudaMemcpyDeviceToHost);
  check_cuda_error();
  cudaFree(device_graph);
}

template<typename Number>
__host__ void floyd_warshall_cuda(const Number *adjancencyMatrix, Number **distanceMatrix, int n) {

  Number *device_graph;
  const size_t size = sizeof(Number) * n * n;
  cudaMalloc(&device_graph, size);
  cudaMemcpy(device_graph, adjancencyMatrix, size, cudaMemcpyHostToDevice);

  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
                (n + block_dim.y - 1) / block_dim.y);

  for (int k = 0; k < n; k++) {
    floyd_warshall_kernel<Number> <<<grid_dim, block_dim>>>(n, k, device_graph);
    cudaThreadSynchronize();
  }

  cudaMemcpy(*distanceMatrix, device_graph, size, cudaMemcpyDeviceToHost);
  check_cuda_error();
  cudaFree(device_graph);
}

template<typename Number>
__host__ void floyd_warshall_cuda(const Number *adjancencyMatrix, Number **distanceMatrix, int **successorMatrix, int n) {

  Number *device_graph;

  const size_t size = sizeof(Number) * n * n;

  cudaMalloc(&device_graph, size);

  cudaMemcpy(device_graph, adjancencyMatrix, size, cudaMemcpyHostToDevice);

  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
                (n + block_dim.y - 1) / block_dim.y);

  for (int k = 0; k < n; k++) {
    floyd_warshall_kernel<Number> <<<grid_dim, block_dim>>>(n, k, device_graph);
    cudaThreadSynchronize();
  }

  cudaMemcpy(*distanceMatrix, device_graph, size, cudaMemcpyDeviceToHost);
  check_cuda_error();
  cudaFree(device_graph);
}

__host__ void show_cuda_device_properties() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);

    std::cout << "Device " << i << ": " << deviceProps.name << "\n"
              << "\tSMs: " << deviceProps.multiProcessorCount << "\n"
              << "\tGlobal mem: " << static_cast<double>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024) << "GB \n"
              << "\tCUDA Cap: " << deviceProps.major << "." << deviceProps.minor << "\n";
  }
}

__host__ void floyd_warshall_blocked_cuda_double(const double *adjancencyMatrix, double **distanceMatrix, int n) {
  floyd_warshall_blocked_cuda<double>(adjancencyMatrix, distanceMatrix, n);
}
__host__ void floyd_warshall_blocked_cuda_double(const double *adjancencyMatrix, double **distanceMatrix, int **successorMatrix, int n) {
  floyd_warshall_blocked_cuda<double>(adjancencyMatrix, distanceMatrix, successorMatrix, n);
}
__host__ void floyd_warshall_blocked_cuda_float(const float *adjancencyMatrix, float **distanceMatrix, int n) {
  floyd_warshall_blocked_cuda<float>(adjancencyMatrix, distanceMatrix, n);
}
__host__ void floyd_warshall_blocked_cuda_float(const float *adjancencyMatrix, float **distanceMatrix, int **successorMatrix, int n) {
  floyd_warshall_blocked_cuda<float>(adjancencyMatrix, distanceMatrix, successorMatrix, n);
}
__host__ void floyd_warshall_blocked_cuda_int(const int *adjancencyMatrix, int **distanceMatrix, int n) {
  floyd_warshall_blocked_cuda<int>(adjancencyMatrix, distanceMatrix, n);
}
__host__ void floyd_warshall_blocked_cuda_int(const int *adjancencyMatrix, int **distanceMatrix, int **successorMatrix, int n) {
  floyd_warshall_blocked_cuda<int>(adjancencyMatrix, distanceMatrix, successorMatrix, n);
}

__host__ void floyd_warshall_cuda_double(const double *adjancencyMatrix, double **distanceMatrix, int n) {
  floyd_warshall_cuda<double>(adjancencyMatrix, distanceMatrix, n);
}
__host__ void floyd_warshall_cuda_double(const double *adjancencyMatrix, double **distanceMatrix, int **successorMatrix, int n) {
  floyd_warshall_cuda<double>(adjancencyMatrix, distanceMatrix, successorMatrix, n);
}
__host__ void floyd_warshall_cuda_float(const float *adjancencyMatrix, float **distanceMatrix, int n) {
  floyd_warshall_cuda<float>(adjancencyMatrix, distanceMatrix, n);
}
__host__ void floyd_warshall_cuda_float(const float *adjancencyMatrix, float **distanceMatrix, int **successorMatrix, int n) {
  floyd_warshall_cuda<float>(adjancencyMatrix, distanceMatrix, successorMatrix, n);
}
__host__ void floyd_warshall_cuda_int(const int *adjancencyMatrix, int **distanceMatrix, int n) {
  floyd_warshall_cuda<int>(adjancencyMatrix, distanceMatrix, n);
}
__host__ void floyd_warshall_cuda_int(const int *adjancencyMatrix, int **distanceMatrix, int **successorMatrix, int n) {
  floyd_warshall_cuda<int>(adjancencyMatrix, distanceMatrix, successorMatrix, n);
}
