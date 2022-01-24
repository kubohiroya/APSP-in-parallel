#pragma once

#ifdef CUDA
template<typename Number> void floyd_warshall_blocked_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int n);
template<typename Number> void floyd_warshall_blocked_successor_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int ** successorMatrix, int n);
template<typename Number> void floyd_warshall_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int n);
template<typename Number> void floyd_warshall_successor_cuda(const Number *adjacencyMatrix, Number** distanceMatrix, int **successorMatrix, int n);

#endif
