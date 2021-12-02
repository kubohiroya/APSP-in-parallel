export void floyd_warshall_in_place_double(uniform double C[], const uniform double A[], const uniform double B[], uniform int successorMatrix[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized) {
  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (C[i * n_oversized + j] > A[i * n_oversized + k] + B[k * n_oversized + j]) {
	    C[i * n_oversized + j] = A[i * n_oversized + k] + B[k * n_oversized + j];
	    if(jb + j < n && ib + i < n && kb + k < n){
          successorMatrix[(jb + j) * n + ib + i] = successorMatrix[(jb + j) * n + kb + k];
        }
      }
    }
  }
}
