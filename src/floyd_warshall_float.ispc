export void floyd_warshall_in_place_float(uniform float C[], const uniform float A[], const uniform float B[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized, const uniform float inf) {

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (A[i * n_oversized + k] != inf && B[k * n_oversized + j] != inf) {
        float sum = A[i * n_oversized + k] + B[k * n_oversized + j];
        if (C[i * n_oversized + j] > sum) {
   	      C[i * n_oversized + j] = sum;
        }
      }
    }
  }
}

export void floyd_warshall_in_place_successor_float(uniform float C[], const uniform float A[], const uniform float B[], uniform int successorMatrix[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized, const uniform float inf) {

  uniform int32 *s1 = &successorMatrix[jb * n + ib];
  uniform int32 *s2 = &successorMatrix[jb * n + kb];

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      /*
      if (C[i * n_oversized + j] > A[i * n_oversized + k] + B[k * n_oversized + j]) {
	    C[i * n_oversized + j] = A[i * n_oversized + k] + B[k * n_oversized + j];
	    if(jb + j < n && ib + i < n && kb + k < n){
          //successorMatrix[(jb + j) * n + ib + i] = successorMatrix[(jb + j) * n + kb + k];
	      successorMatrix[jbib + j * n + i] = successorMatrix[jbkb + j * n + k];
        }
      }
      */
      if (A[i * n_oversized + k] != inf && B[k * n_oversized + j] != inf) {
        float sum = A[i * n_oversized + k] + B[k * n_oversized + j];
        if (C[i * n_oversized + j] > sum) {
  	      C[i * n_oversized + j] = sum;
          s1[i + j * n] = s2[k + j * n];
        }
      }
    }
  }
}
