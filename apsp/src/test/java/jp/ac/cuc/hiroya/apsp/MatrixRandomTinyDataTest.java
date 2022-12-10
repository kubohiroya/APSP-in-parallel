package jp.ac.cuc.hiroya.apsp;

import org.junit.jupiter.api.Test;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;

public class MatrixRandomTinyDataTest {

    static int NUM_BLOCKS = -1;
    static String ENV = "omp-ispc";

    interface tinyRandomMatrixDouble {
        long seed = 0;
        int n = 5;
        double p = 0.5;
        double min = 1.0;
        double max = 100.0;
    }

    interface tinyRandomMatrixInt {
        long seed = 0;
        int n = 5;
        double p = 0.5;
        int min = 1;
        int max = 100;
    }

    String getRandomTinyMatrixDouble() {
        long seed = tinyRandomMatrixDouble.seed;
        int n = tinyRandomMatrixDouble.n;
        double p = tinyRandomMatrixDouble.p;
        double min = tinyRandomMatrixDouble.min;
        double max = tinyRandomMatrixDouble.max;
        return "random:" + seed + "," + n + "," + p + "," + min + "," + max;
    }

    String getRandomTinyMatrixInt() {
        long seed = tinyRandomMatrixInt.seed;
        int n = tinyRandomMatrixInt.n;
        double p = tinyRandomMatrixInt.p;
        int min = tinyRandomMatrixInt.min;
        int max = tinyRandomMatrixInt.max;
        return "random:" + seed + "," + n + "," + p + "," + min + "," + max;
    }

    @Test
    public void 小さめIntランダムデータの同じシードでの生成内容が同じ() {
        long seed = tinyRandomMatrixInt.seed;
        int n = tinyRandomMatrixInt.n;
        double p = tinyRandomMatrixInt.p;
        int min = tinyRandomMatrixInt.min;
        int max = tinyRandomMatrixInt.max;

        int[] matrix1 = RandomMatrixGenerator.generateRandomAdjacencyMatrix(
                seed, n, p, min, max
        );
        int[] matrix2 = RandomMatrixGenerator.generateRandomAdjacencyMatrix(
                seed, n, p, min, max
        );
        int[] matrix3 = RandomMatrixGenerator.generateRandomAdjacencyMatrix(
                seed, n, p, min, max
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
    public void 小さめDoubleランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        MatrixAssertion.assertDistancesBetweenAlgorithmsDouble(getRandomTinyMatrixDouble(), null, null,
                new String[][]{{ENV, "f"}, {ENV, "j"}}, new int[]{-1, -1}, false);
    }

    @Test
    public void 小さめIntランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        MatrixAssertion.assertDistancesBetweenAlgorithmsInt(getRandomTinyMatrixInt(), null, null,
                new String[][]{{ENV, "f"}, {ENV, "j"}}, new int[]{-1, -1}, false);
    }
}
