package jp.ac.cuc.hiroya.apsp.lib;

public interface ApspResolver<T> {
    ApspResult<T> resolveWithJohnson(String execEnv, T input, int v);
    ApspResult<T> resolveWithFloydWarshall(String execEnv, T input, int v);
    ApspResult<T> resolveWithFloydWarshall(String execEnv, T input, int v, int numBlocks);
    ApspResult<T> resolve(String execEnv, String algorithm, T input, int v, int numBlocks);
    T getInfinity(String execEnv);

    interface EXEC_ENV {
        String SEQ = "seq";
        String SEQ_ISPC = "seq-ispc";
        String OMP = "omp";
        String OMP_ISPC = "omp-ispc";
        String CUDA = "cuda";
    }

    interface ALGORITHM {
        String FLOYD_WARSHALL = "Floyd-Warshall";
        String F = "f";
        String JOHNSON = "Johnson";
        String J = "j";
        int FLOYD_WARSHALL_BLOCK_SIZE = 96;
    }

}
