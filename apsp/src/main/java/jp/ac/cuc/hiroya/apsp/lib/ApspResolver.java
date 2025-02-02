package jp.ac.cuc.hiroya.apsp.lib;

public interface ApspResolver<T> {
    ApspResult<T> resolveWithJohnson(String execEnv, T adjacencyMatrix);
    ApspResult<T> resolve(String execEnv, int v, int e,
                          int[] edges,
                          T distances);
    ApspResult<T> resolveWithFloydWarshall(String execEnv, T adjacencyMatrix);
    ApspResult<T> resolveWithFloydWarshall(String execEnv, T adjacencyMatrix, int numBlocks);
    ApspResult<T> resolve(String execEnv, String algorithm, T adjacencyMatrix, int numBlocks);
    T getInfinity(String execEnv);

    public interface EXEC_ENV {
        String SEQ = "seq";
        String SEQ_ISPC = "seq-ispc";
        String OMP = "omp";
        String OMP_ISPC = "omp-ispc";
        String CUDA = "cuda";
    }

    public interface ALGORITHM {
        String FLOYD_WARSHALL = "Floyd-Warshall";
        String F = "f";
        String JOHNSON = "Johnson";
        String J = "j";
        int FLOYD_WARSHALL_BLOCK_SIZE = 16;
    }

}
