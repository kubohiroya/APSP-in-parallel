export void floyd_warshall_in_place_double(uniform double C[], const uniform double A[], const uniform double B[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized, const uniform double inf) {

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (A[i * n_oversized + k] != inf && B[k * n_oversized + j] != inf) {
        double sum = A[i * n_oversized + k] + B[k * n_oversized + j];
        if (C[i * n_oversized + j] > sum) {
	  C[i * n_oversized + j] = sum;
        }
      }
    }
  }
}

export void floyd_warshall_in_place_successor_double(uniform double C[], const uniform double A[], const uniform double B[], uniform int successorMatrix[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized, const uniform double inf) {

  uniform int32 *s1 = &successorMatrix[jb * n + ib];
  uniform int32 *s2 = &successorMatrix[jb * n + kb];

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (A[i * n_oversized + k] != inf && B[k * n_oversized + j] != inf) {
        double sum = A[i * n_oversized + k] + B[k * n_oversized + j];
        if (C[i * n_oversized + j] > sum) {
  	  C[i * n_oversized + j] = sum;
	  s1[j * n + i] = s2[j * n + k];
        }
      }
    }
  }
}

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
      if (A[i * n_oversized + k] != inf && B[k * n_oversized + j] != inf) {
        float sum = A[i * n_oversized + k] + B[k * n_oversized + j];
        if (C[i * n_oversized + j] > sum) {
   	  C[i * n_oversized + j] = sum;
	  s1[j * n + i] = s2[j * n + k];
        }
      }
    }
  }
}

export void floyd_warshall_in_place_int(uniform int C[], const uniform int A[], const uniform int B[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized, const uniform int inf) {

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (A[i * n_oversized + k] != inf && B[k * n_oversized + j] != inf) {
        int sum = A[i * n_oversized + k] + B[k * n_oversized + j];
        if (C[i * n_oversized + j] > sum) {
          C[i * n_oversized + j] = sum;
        }
      }
    }
  }
}
export void floyd_warshall_in_place_successor_int(uniform int C[], const uniform int A[], const uniform int B[], uniform int successorMatrix[],
				    const uniform int kb, const uniform int ib, const uniform int jb,
				    const uniform int b, const uniform int n, const uniform int n_oversized, const uniform int inf) {

  uniform int32 *s1 = &successorMatrix[jb * n + ib];
  uniform int32 *s2 = &successorMatrix[jb * n + kb];

  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (A[i * n_oversized + k] != inf && B[k * n_oversized + j] != inf) {
        int sum = A[i * n_oversized + k] + B[k * n_oversized + j];
        if (C[i * n_oversized + j] > sum) {
          C[i * n_oversized + j] = sum;
	  s1[j * n + i] = s2[j * n + k];
        }
      }
    }
  }
}
