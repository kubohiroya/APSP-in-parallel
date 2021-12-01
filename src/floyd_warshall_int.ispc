export void floyd_warshall_in_place_int(uniform int C[], const uniform int A[], const uniform int B[], uniform int successorMatrix[],
				    const uniform int _kb, const uniform int _ib, const uniform int _jb,
				    const uniform int b, const uniform int n) {
  for (int k = 0; k < b; k++) {
    foreach (i = 0 ... b, j = 0 ... b) {
      if (C[i*n + j] > A[i*n + k] + B[k*n + j]) {
	    C[i*n + j] = A[i*n + k] + B[k*n + j];
	    if(_jb + j < n && _ib + i < n && _kb + k < n){
          successorMatrix[(_jb + j) * n + _ib + i] = successorMatrix[(_jb + j) * n + _kb + k];
        }
      }
    }
  }
}
