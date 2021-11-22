package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;


public class MatrixRandomLargeDataTest {

    static int NUM_BLOCKS = -1;
    static String ENV = "omp-ispc";

    interface largeRandomMatrixDouble{
        long seed = 0;
        int n = 512;
        double p = 0.4;
        double min = 1.0;
        double max = 100.0;
    }

    interface largeRandomMatrixInt{
        long seed = 0;
        int n = 512;
        double p = 0.4;
        int min = 1;
        int max = 100;
    }

    String getRandomLargeMatrixDouble(){
        long seed = largeRandomMatrixDouble.seed;
        int n = largeRandomMatrixDouble.n;
        double p = largeRandomMatrixDouble.n;
        double min = largeRandomMatrixDouble.min;
        double max = largeRandomMatrixDouble.max;
        double inf = Infinity.DBL_INF;
        return "random:"+seed+","+n+","+p+","+min+","+max+",inf";
    }

    String getRandomLargeMatrixInt(){
        long seed = largeRandomMatrixInt.seed;
        int n = largeRandomMatrixInt.n;
        double p = largeRandomMatrixInt.p;
        int min = largeRandomMatrixInt.min;
        int max = largeRandomMatrixInt.max;
        double inf = Infinity.INT_INF;
        return "random:"+seed+","+n+","+p+","+min+","+max+",inf";
    }

    @Test
    public void 大きめIntランダムデータの同じシードでの生成内容が同じ(){
        long seed = largeRandomMatrixInt.seed;
        int n = largeRandomMatrixInt.n;
        double p = largeRandomMatrixInt.p;
        int min = largeRandomMatrixInt.min;
        int max = largeRandomMatrixInt.max;
        int inf = Infinity.INT_INF;

        int[] matrix1 = RandomMatrixGenerator.generateRandomAdjacencyMatrix(
                seed, n, p, min, max, inf
        );
        int[] matrix2 = RandomMatrixGenerator.generateRandomAdjacencyMatrix(
                seed, n, p, min, max, inf
        );
        int[] matrix3 = RandomMatrixGenerator.generateRandomAdjacencyMatrix(
                seed, n, p, min, max, inf
        );
        assertThat(matrix1, is(matrix2));
        assertThat(matrix1, is(matrix3));
    }

    @Test
    public void 大きめDoubleランダムデータのFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomLargeMatrixDouble(), null, null, execEnv, algorithm, NUM_BLOCKS, true);
    }

    @Test
    public void 大きめDoubleランダムデータのJohnson法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomLargeMatrixDouble(), null, null, execEnv, algorithm, NUM_BLOCKS, false);
    }

    @Test
    public void 大きめDoubleランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        MatrixAssertion.assertDistancesBetweenAlgorithmsDouble(getRandomLargeMatrixDouble(), null, null,
                new String[][] {{ENV, "f"},{ENV, "j"}},  NUM_BLOCKS, true);
    }

    @Test
    public void 大きめIntランダムデータのFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithSelfDataInt(getRandomLargeMatrixInt(), null, null, execEnv, algorithm, NUM_BLOCKS, false);
    }

    @Test
    public void 大きめIntランダムデータのJohnson法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithSelfDataInt(getRandomLargeMatrixInt(), null, null, execEnv, algorithm, NUM_BLOCKS, false);
    }

    @Test
    public void 大きめIntランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        MatrixAssertion.assertDistancesBetweenAlgorithmsInt(getRandomLargeMatrixInt(), null, null,
                new String[][] {{ENV, "f"},{ENV, "j"}}, NUM_BLOCKS,  true);
    }
}
