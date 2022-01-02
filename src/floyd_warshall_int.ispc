export void floyd_warshall_in_place_int(uniform int C[], const uniform int A[], const uniform int B[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized) {

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      int sum = A[i * n_oversized + k] + B[k * n_oversized + j];
      if (C[i * n_oversized + j] > sum) {
        C[i * n_oversized + j] = sum;
       }
    }
  }
}
export void floyd_warshall_in_place_successor_int(uniform int C[], const uniform int A[], const uniform int B[], uniform int successorMatrix[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized) {

  uniform int32 *s1 = &successorMatrix[jb * n + ib];
  uniform int32 *s2 = &successorMatrix[jb * n + kb];

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      int sum = A[i * n_oversized + k] + B[k * n_oversized + j];
      if (C[i * n_oversized + j] > sum) {
        C[i * n_oversized + j] = sum;
        s1[i + j * n] = s2[k + j * n];
      }
    }
  }
}
