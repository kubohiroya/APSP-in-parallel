 #include <iostream>
 #include <stdio.h>
// #include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "util.hpp"
#include "getInf.hpp"
#include "floyd_warshall.cuh"

#define BLOCK_SIZE 16

/**
 * CUDA handle error, if error occurs print message and exit program
*
* @param error: CUDA error status
*/
#define HANDLE_ERROR(error) { \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} \

/*
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
*/

/**
 * Naive CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * check if path from vertex x -> y will be short using vertex u x -> u -> y
 * for all vertices in graph
 *
 * @param u: Index of vertex u
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
template<typename T>
static __global__
void _naive_fw_kernel(const int u, size_t pitch, size_t intPitch, const int nvertex, T* const graph, int* const pred, T const inf) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < nvertex && x < nvertex) {
        int indexUX = u * pitch + x;
        int indexYU = y * pitch + u;
        if(graph[indexYU] == inf || graph[indexUX] == inf){
	  return;
	}
        int indexYX = y * pitch + x;
        T newPath = graph[indexYU] + graph[indexUX];
        T oldPath = graph[indexYX];
        if (oldPath == inf || oldPath > newPath) {
            graph[indexYX] = newPath;
            pred[y * intPitch + x] = pred[u * intPitch + x];
        }
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Dependent phase 1
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
template<typename T>
static __global__
void _blocked_fw_dependent_ph(const int blockId, size_t pitch, size_t intPitch, const int nvertex, T* const graph, int* const pred, T const inf) {
    __shared__ T cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePred[BLOCK_SIZE][BLOCK_SIZE];

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = BLOCK_SIZE * blockId + idy;
    const int v2 = BLOCK_SIZE * blockId + idx;

    int newPred;
    T newPath;

    const int cellId = v1 * pitch + v2;
    const int predCellId = v1 * intPitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        cacheGraph[idy][idx] = graph[cellId];
        cachePred[idy][idx] = pred[predCellId];
        newPred = cachePred[idy][idx];
    } else {
        cacheGraph[idy][idx] = inf;
        cachePred[idy][idx] = -1;
    }

    // Synchronize to make sure the all value are loaded in block
    __syncthreads();

    #pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        if(cacheGraph[idy][u] == inf || cacheGraph[u][idx] == inf){
	    newPath = inf;
	} else {
            newPath = cacheGraph[idy][u] + cacheGraph[u][idx];
	}

        // Synchronize before calculate new value
        __syncthreads();
        if (newPath < cacheGraph[idy][idx]) {
            cacheGraph[idy][idx] = newPath;
            newPred = cachePred[u][idx];
        }

        // Synchronize to make sure that all value are current
        __syncthreads();
        cachePred[idy][idx] = newPred;
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = cacheGraph[idy][idx];
        pred[predCellId] = cachePred[idy][idx];
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Partial dependent phase 2
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
template<typename T>
static __global__
void _blocked_fw_partial_dependent_ph(const int blockId, size_t pitch, size_t intPitch, const int nvertex, T* const graph, int* const pred, T const inf) {
    if (blockIdx.x == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    int v1 = BLOCK_SIZE * blockId + idy;
    int v2 = BLOCK_SIZE * blockId + idx;

    __shared__ T cacheGraphBase[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePredBase[BLOCK_SIZE][BLOCK_SIZE];

    // Load base block for graph and predecessors
    int cellId = v1 * pitch + v2;
    int predCellId = v1 * intPitch + v2;

    if (v1 < nvertex && v2 < nvertex) {
        cacheGraphBase[idy][idx] = graph[cellId];
        cachePredBase[idy][idx] = pred[predCellId];
    } else {
        cacheGraphBase[idy][idx] = inf;
        cachePredBase[idy][idx] = -1;
    }

    // Load i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
        v2 = BLOCK_SIZE * blockIdx.x + idx;
    } else {
    // Load j-aligned singly dependent blocks
        v1 = BLOCK_SIZE * blockIdx.x + idy;
    }

    __shared__ T cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePred[BLOCK_SIZE][BLOCK_SIZE];

    // Load current block for graph and predecessors
    T currentPath;
    int currentPred;

    cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        currentPath = graph[cellId];
        currentPred = pred[predCellId];
    } else {
        currentPath = inf;
        currentPred = -1;
    }
    cacheGraph[idy][idx] = currentPath;
    cachePred[idy][idx] = currentPred;

    // Synchronize to make sure the all value are saved in cache
    __syncthreads();

    T newPath;
    // Compute i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            if (cacheGraphBase[idy][u] == inf || cacheGraph[u][idx] == inf) {
              newPath = inf;
	    } else {
	      newPath = cacheGraphBase[idy][u] + cacheGraph[u][idx];
	    }

            if (newPath < currentPath) {
                currentPath = newPath;
                currentPred = cachePred[u][idx];
            }
            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

           // Update new values
            cacheGraph[idy][idx] = currentPath;
            cachePred[idy][idx] = currentPred;

           // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    } else {
    // Compute j-aligned singly dependent blocks
        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            if (cacheGraphBase[idy][u] == inf || cacheGraph[u][idx] == inf) {
                newPath = inf;
	    } else {
                newPath = cacheGraph[idy][u] + cacheGraphBase[u][idx];
	    }

            if (newPath < currentPath) {
                currentPath = newPath;
                currentPred = cachePredBase[u][idx];
            }

            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

           // Update new values
            cacheGraph[idy][idx] = currentPath;
            cachePred[idy][idx] = currentPred;

           // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = currentPath;
        pred[predCellId] = currentPred;
    }
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Independent phase 3
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
template<typename T>
static __global__
void _blocked_fw_independent_ph(const int blockId, size_t pitch, size_t intPitch, const int nvertex, T* const graph, int* const pred, T const inf) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = blockDim.y * blockIdx.y + idy;
    const int v2 = blockDim.x * blockIdx.x + idx;

    __shared__ T cacheGraphBaseRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T cacheGraphBaseCol[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cachePredBaseRow[BLOCK_SIZE][BLOCK_SIZE];

    int v1Row = BLOCK_SIZE * blockId + idy;
    int v2Col = BLOCK_SIZE * blockId + idx;

    // Load data for block
    int cellId;
    int predCellId;

    if (v1Row < nvertex && v2 < nvertex) {
        cellId = v1Row * pitch + v2;
        predCellId = v1Row * intPitch + v2;

        cacheGraphBaseRow[idy][idx] = graph[cellId];
        cachePredBaseRow[idy][idx] = pred[predCellId];
    }
    else {
        cacheGraphBaseRow[idy][idx] = inf;
        cachePredBaseRow[idy][idx] = -1;
    }

    if (v1  < nvertex && v2Col < nvertex) {
        cellId = v1 * pitch + v2Col;
        cacheGraphBaseCol[idy][idx] = graph[cellId];
    }
    else {
        cacheGraphBaseCol[idy][idx] = inf;
    }

    // Synchronize to make sure the all value are loaded in virtual block
   __syncthreads();

   T currentPath;
   int currentPred;
   T newPath;

   // Compute data for block
   if (v1  < nvertex && v2 < nvertex) {
       cellId = v1 * pitch + v2;
       predCellId = v1 * intPitch + v2;

       currentPath = graph[cellId];
       currentPred = pred[cellId];

        #pragma unroll
       for (int u = 0; u < BLOCK_SIZE; ++u) {
           if (cacheGraphBaseCol[idy][u] == inf || cacheGraphBaseRow[u][idx] == inf) {
               newPath = inf;
           } else {
               newPath = cacheGraphBaseCol[idy][u] + cacheGraphBaseRow[u][idx];
	   }
           if (currentPath > newPath) {
               currentPath = newPath;
               currentPred = cachePredBaseRow[u][idx];
           }
       }
       graph[cellId] = currentPath;
       pred[predCellId] = currentPred;
   }
}

/**
 * Allocate memory on device and copy memory from host to device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param graphDevice: Pointer to array of graph with distance between vertex on device
 * @param predDevice: Pointer to array of predecessors for a graph on device
 *
 * @return: Pitch for allocation
 */
template<typename T>
static
void _cudaMoveMemoryToDevice(int nvertex, T *graph, int *pred, T **graphDevice, int **predDevice, size_t *pitch, size_t *intPitch) {
    size_t height = nvertex;
    size_t intWidth = height * sizeof(int);
    size_t width = height * sizeof(T);

    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    HANDLE_ERROR(cudaMallocPitch(graphDevice, pitch, width, height));
    HANDLE_ERROR(cudaMallocPitch(predDevice, intPitch, intWidth, height));
    // Copy input from host memory to GPU buffers and
    HANDLE_ERROR(cudaMemcpy2D(*graphDevice, *pitch,
			      graph, width, width, height, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(*predDevice, *intPitch,
            pred, intWidth, intWidth, height, cudaMemcpyHostToDevice));

}

/**
 * Copy memory from device to host and free device memory
 *
 * @param graphDevice: Array of graph with distance between vertex on device
 * @param predDevice: Array of predecessors for a graph on device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param pitch: Pitch for allocation
 */
template<typename T>
static
void _cudaMoveMemoryToHost(int nvertex, T *graph, int *pred, T *graphDevice, int *predDevice, size_t pitch, size_t intPitch) {
    size_t height = nvertex;
    size_t intWidth = height * sizeof(int);
    size_t width = height * sizeof(T);

    HANDLE_ERROR(cudaMemcpy2D(pred, intWidth, predDevice, intPitch, intWidth, height, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy2D(graph, width, graphDevice, pitch, width, height, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(predDevice));
    HANDLE_ERROR(cudaFree(graphDevice));
}

/**
 * Naive implementation of Floyd Warshall algorithm in CUDA
 *
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 */
template<typename T>
void cudaNaiveFW(int nvertex, T *graph, int *pred) {
    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));

    // Initialize the grid and block dimensions here
    dim3 dimGrid((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    T *graphDevice;
    int *predDevice;
    size_t pitch, intPitch;
    _cudaMoveMemoryToDevice(nvertex, graph, pred, &graphDevice, &predDevice, &pitch, &intPitch);
    T inf = getInf<T>();

    // cudaFuncSetCacheConfig(_naive_fw_kernel, cudaFuncCachePreferL1);
    for(int vertex = 0; vertex < nvertex; ++vertex) {
      _naive_fw_kernel<<<dimGrid, dimBlock>>>(vertex, pitch / sizeof(T), intPitch / sizeof(int), nvertex, graphDevice, predDevice, inf);
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _cudaMoveMemoryToHost(nvertex, graph, pred, graphDevice, predDevice,  pitch, intPitch);
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
template<typename T>
void cudaBlockedFW(int nvertex, T *graph, int *pred) {
    HANDLE_ERROR(cudaSetDevice(0));
    T *graphDevice;
    int *predDevice;
    size_t pitch, intPitch;
    _cudaMoveMemoryToDevice(nvertex, graph, pred, &graphDevice, &predDevice, &pitch, &intPitch);

    dim3 gridPhase1(1 ,1, 1);
    dim3 gridPhase2((nvertex - 1) / BLOCK_SIZE + 1, 2 , 1);
    dim3 gridPhase3((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1 , 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

    int numBlock = (nvertex - 1) / BLOCK_SIZE + 1;
    T inf = getInf<T>();

    for(int blockID = 0; blockID < numBlock; ++blockID) {
        // Start dependent phase
        _blocked_fw_dependent_ph<<<gridPhase1, dimBlockSize>>>
	  (blockID, pitch / sizeof(T), intPitch / sizeof(int), nvertex, graphDevice, predDevice, inf);

        // Start partially dependent phase
        _blocked_fw_partial_dependent_ph<<<gridPhase2, dimBlockSize>>>
	  (blockID, pitch / sizeof(T), intPitch / sizeof(int), nvertex, graphDevice, predDevice, inf);

        // Start independent phase
        _blocked_fw_independent_ph<<<gridPhase3, dimBlockSize>>>
	  (blockID, pitch / sizeof(T), intPitch / sizeof(int), nvertex, graphDevice, predDevice, inf);
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _cudaMoveMemoryToHost(nvertex, graph, pred, graphDevice, predDevice, pitch, intPitch);
}


template<typename T>
__host__ void floyd_warshall_blocked_cuda(const T *adjancencyMatrix, T **distanceMatrix, int nvertex) {
  *distanceMatrix = (T *)malloc(sizeof(T) * nvertex * nvertex);
  memcpy(*distanceMatrix, adjancencyMatrix, sizeof(T) * nvertex * nvertex);
  int* successorMatrix;
  successorMatrix = (int*) malloc(sizeof(int) * nvertex * nvertex);
  memset(successorMatrix, 0, sizeof(int) * nvertex * nvertex);
  cudaBlockedFW<T>(nvertex, *distanceMatrix, successorMatrix);
  delete[] successorMatrix;
}

template<typename T>
__host__ void floyd_warshall_blocked_successor_cuda(const T *adjancencyMatrix, T **distanceMatrix, int **successorMatrix, int nvertex) {
  *distanceMatrix = (T *)malloc(sizeof(T) * nvertex * nvertex);
  memcpy(*distanceMatrix, adjancencyMatrix, sizeof(T) * nvertex * nvertex);
  *successorMatrix = (int *)malloc(sizeof(int) * nvertex * nvertex);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i< nvertex; i++) {
    for (unsigned int j = 0; j< nvertex; j++) {
      (*successorMatrix)[i * nvertex + j] = i;
    }
  }
  cudaBlockedFW<T>(nvertex, *distanceMatrix, *successorMatrix);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i< nvertex; i++) {
    for (unsigned int j = i; j< nvertex; j++) {
      int index1 = i * nvertex + j;
      int index2 = j * nvertex + i;
      T value =(*successorMatrix)[index1];
      (*successorMatrix)[index1] = (*successorMatrix)[index2];
      (*successorMatrix)[index2] = value;
    }
  }
}

template<typename T>
__host__ void floyd_warshall_cuda(const T *adjancencyMatrix, T **distanceMatrix, int nvertex) {
  *distanceMatrix = (T *)malloc(sizeof(T) * nvertex * nvertex);
  memcpy(*distanceMatrix, adjancencyMatrix, sizeof(T) * nvertex * nvertex);
  int* successorMatrix;
  successorMatrix = (int*) malloc(sizeof(int) * nvertex * nvertex);
  memset(successorMatrix, 0, sizeof(int) * nvertex * nvertex);
  cudaNaiveFW<T>(nvertex, *distanceMatrix, successorMatrix);
  delete[] successorMatrix;
}

template<typename T>
__host__ void floyd_warshall_successor_cuda(const T *adjancencyMatrix, T **distanceMatrix, int **successorMatrix, int nvertex) {
  *distanceMatrix = (T *)malloc(sizeof(T) * nvertex * nvertex);
  memcpy(*distanceMatrix, adjancencyMatrix, sizeof(T) * nvertex * nvertex);
  *successorMatrix = (int *)malloc(sizeof(int) * nvertex * nvertex);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i< nvertex; i++) {
    for (unsigned int j = 0; j< nvertex; j++) {
      (*successorMatrix)[i * nvertex + j] = i;
    }
  }
  cudaNaiveFW<T>(nvertex, *distanceMatrix, *successorMatrix);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (unsigned int i = 0; i< nvertex; i++) {
    for (unsigned int j = i; j< nvertex; j++) {
      int index1 = i * nvertex + j;
      int index2 = j * nvertex + i;
      T value =(*successorMatrix)[index1];
      (*successorMatrix)[index1] = (*successorMatrix)[index2];
      (*successorMatrix)[index2] = value;
    }
  }
}

template __host__ void floyd_warshall_blocked_cuda<double>(const double *adjancencyMatrix, double **distanceMatrix, const int nvertex);
template __host__ void floyd_warshall_blocked_cuda<float>(const float *adjancencyMatrix, float **distanceMatrix, const int nvertex);
template __host__ void floyd_warshall_blocked_cuda<int>(const int *adjancencyMatrix, int **distanceMatrix, const int nvertex);
template __host__ void floyd_warshall_blocked_successor_cuda<double>(const double *adjancencyMatrix, double **distanceMatrix, int **successorMatrix, const int nvertex);
template __host__ void floyd_warshall_blocked_successor_cuda<float>(const float *adjancencyMatrix, float **distanceMatrix, int **successorMatrix, const int nvertex);
template __host__ void floyd_warshall_blocked_successor_cuda<int>(const int *adjancencyMatrix, int **distanceMatrix, int **successorMatrix, const int nvertex);

template __host__ void floyd_warshall_cuda<double>(const double *adjancencyMatrix, double **distanceMatrix, const int nvertex);
template __host__ void floyd_warshall_cuda<float>(const float *adjancencyMatrix, float **distanceMatrix, const int nvertex);
template __host__ void floyd_warshall_cuda<int>(const int *adjancencyMatrix, int **distanceMatrix, const int nvertex);
template __host__ void floyd_warshall_successor_cuda<double>(const double *adjancencyMatrix, double **distanceMatrix, int **successorMatrix, const int nvertex);
template __host__ void floyd_warshall_successor_cuda<float>(const float *adjancencyMatrix, float **distanceMatrix, int **successorMatrix, const int nvertex);
template __host__ void floyd_warshall_successor_cuda<int>(const int *adjancencyMatrix, int **distanceMatrix, int **successorMatrix, const int nvertex);
