package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;

public class MatrixRandomTinyDataTest {

    static int NUM_BLOCKS = -1;
    static String ENV = "omp-ispc";

    interface tinyRandomMatrixDouble {
        long seed = 0;
        int n = 8;
        double p = 0.4;
        double min = 1.0;
        double max = 100.0;
    }

    interface tinyRandomMatrixInt {
        long seed = 0;
        int n = 8;
        double p = 0.3;
        int min = 1;
        int max = 100;
    }

    String getRandomTinyMatrixDouble() {
        long seed = tinyRandomMatrixDouble.seed;
        int n = tinyRandomMatrixDouble.n;
        double p = tinyRandomMatrixDouble.p;
        double min = tinyRandomMatrixDouble.min;
        double max = tinyRandomMatrixDouble.max;
        return "random:" + seed + "," + n + "," + p + "," + min + "," + max + ",inf";
    }

    String getRandomTinyMatrixInt() {
        long seed = tinyRandomMatrixInt.seed;
        int n = tinyRandomMatrixInt.n;
        double p = tinyRandomMatrixInt.p;
        int min = tinyRandomMatrixInt.min;
        int max = tinyRandomMatrixInt.max;
        double inf = Infinity.INT_INF;
        return "random:" + seed + "," + n + "," + p + "," + min + "," + max + ",inf";
    }

    @Test
    public void 小さめIntランダムデータの同じシードでの生成内容が同じ() {
        long seed = tinyRandomMatrixInt.seed;
        int n = tinyRandomMatrixInt.n;
        double p = tinyRandomMatrixInt.p;
        int min = tinyRandomMatrixInt.min;
        int max = tinyRandomMatrixInt.max;
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
    public void 小さめDoubleランダムデータのFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomTinyMatrixDouble(),
                null, null, execEnv, algorithm, NUM_BLOCKS, false);
    }

    @Test
    public void 小さめDoubleランダムデータのJohnson法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomTinyMatrixDouble(), null, null, execEnv, algorithm, NUM_BLOCKS, false);
    }

    @Test
    public void 小さめDoubleランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        MatrixAssertion.assertDistancesBetweenAlgorithmsDouble(getRandomTinyMatrixDouble(), null, null,
                new String[][]{{ENV, "f"}, {ENV, "j"}}, NUM_BLOCKS, false);
    }

    @Test
    public void 小さめIntランダムデータのFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithSelfDataInt(getRandomTinyMatrixInt(),
                null, null, execEnv, algorithm, NUM_BLOCKS, false);
    }

    @Test
    public void 小さめIntランダムデータのJohnson法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithSelfDataInt(getRandomTinyMatrixInt(), null, null, execEnv, algorithm, NUM_BLOCKS, false);
    }

    @Test
    public void 小さめIntランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        MatrixAssertion.assertDistancesBetweenAlgorithmsInt(getRandomTinyMatrixInt(), null, null,
                new String[][]{{ENV, "f"}, {ENV, "j"}}, NUM_BLOCKS, false);
    }
}
