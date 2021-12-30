export void floyd_warshall_in_place_double(uniform double C[], const uniform double A[], const uniform double B[], uniform int successorMatrix[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized) {
  const uniform int jbib = jb * n + ib;
  const uniform int jbkb = jb * n + kb;
  
  uniform int* s1 = &successorMatrix[jbib];
  const uniform int32 *s2 = &successorMatrix[jbkb];
  
  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      double sum = A[i * n_oversized + k] + B[k * n_oversized + j];
      if (C[i * n_oversized + j] > sum) {
	C[i * n_oversized + j] = sum;
	if(jb + j < n && ib + i < n && kb + k < n){
	  s1[j * n + i] = s2[j * n + k];
        }
      }
    }
  }
}
