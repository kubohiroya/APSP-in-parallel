package jp.ac.cuc.hiroya.apsp;

import org.junit.Test;

import java.text.NumberFormat;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;


public class MatrixRandomLargeDataTest {

    static int NUM_BLOCKS_0 = 4;
    static int NUM_BLOCKS_1 = 5;

    static String ENV = "omp-ispc";

    interface largeRandomMatrixDouble{
        long seed = 10;
        int n = 12;
        double p = 0.4;
        double min = 1.0;
        double max = 100.0;
    }

    interface largeRandomMatrixInt{
        long seed = 10;
        int n = 10000;
        double p = 0.001;
        int min = 1;
        int max = 100;
    }

    String getRandomLargeMatrixDouble(int n){
        long seed = largeRandomMatrixDouble.seed;
        double p = largeRandomMatrixDouble.n;
        double min = largeRandomMatrixDouble.min;
        double max = largeRandomMatrixDouble.max;
        return "random:"+seed+","+n+","+p+","+min+","+max;
    }

    String getRandomLargeMatrixInt(int n){
        long seed = largeRandomMatrixInt.seed;
        double p = largeRandomMatrixInt.p;
        int min = largeRandomMatrixInt.min;
        int max = largeRandomMatrixInt.max;
        return "random:"+seed+","+n+","+p+","+min+","+max;
    }

    @Test
    public void 大きめIntランダムデータの同じシードでの生成内容が同じ(){
        long seed = largeRandomMatrixInt.seed;
        int n = largeRandomMatrixInt.n;
        double p = largeRandomMatrixInt.p;
        int min = largeRandomMatrixInt.min;
        int max = largeRandomMatrixInt.max;

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
    public void 大きめDoubleランダムデータを余り有りブロックFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        int n = largeRandomMatrixDouble.n;
        MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomLargeMatrixDouble(n), null, null, execEnv, algorithm, NUM_BLOCKS_1, false);
    }

    @Test
    public void 大きめIntランダムデータを余り有りブロックFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        int n = largeRandomMatrixInt.n;
        MatrixAssertion.assertDistancesWithSelfDataInt(getRandomLargeMatrixInt(n), null, null, execEnv, algorithm, NUM_BLOCKS_1, false);
    }

    @Test
    public void 大きめDoubleランダムデータを余り無しブロックFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        int n = largeRandomMatrixDouble.n;
        MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomLargeMatrixDouble(n), null, null, execEnv, algorithm, NUM_BLOCKS_0, false);
    }

    @Test
    public void 大きめIntランダムデータを余り無しブロックFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "f";
        int n = largeRandomMatrixInt.n;
        MatrixAssertion.assertDistancesWithSelfDataInt(getRandomLargeMatrixInt(n), null, null, execEnv, algorithm, NUM_BLOCKS_0, false);
    }

    @Test
    public void 大きめIntランダムデータのJohnson法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "j";
        int n = largeRandomMatrixInt.n;
        MatrixAssertion.assertDistancesWithSelfDataInt(getRandomLargeMatrixInt(n), null, null, execEnv, algorithm, -1, false);
    }

    @Test
    public void 大きめDoubleランダムデータのJohnson法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "j";
        int n = largeRandomMatrixDouble.n;
        MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomLargeMatrixDouble(n), null, null, execEnv, algorithm, -1, false);
    }

    @Test
    public void 大きめDoubleランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        int n = 99;
        MatrixAssertion.assertDistancesBetweenAlgorithmsDouble(getRandomLargeMatrixDouble(n), null, null,
                new String[][] {{ENV, "f"}, {ENV, "f"}, {ENV, "f"}, {ENV, "f"}}, new int[]{-1, 3, 4, 5}, false);
    }

    @Test
    public void 大きめIntランダムデータのアルゴリズム間での処理結果の整合性を相互検証() throws Exception {
        int n = 99;
        MatrixAssertion.assertDistancesBetweenAlgorithmsInt(getRandomLargeMatrixInt(n), null, null,
                new String[][] {{ENV, "f"}, {ENV, "f"}, {ENV, "f"}, {ENV, "f"}}, new int[]{-1, 3, 4, 5},  false);
    }

    @Test
    public void 本番データのJohnson法でのメモリリークをしているかどうかの検証() throws Exception {
        String execEnv = ENV;
        String algorithm = "j";
        int n = largeRandomMatrixDouble.n;

        long total = Runtime.getRuntime().totalMemory();
        long max = Runtime.getRuntime().maxMemory();
        NumberFormat nfNum = NumberFormat.getNumberInstance();

        System.out.println("total " + nfNum.format(total / 1024 / 1024) + " MB");
        System.out.println("max   " + nfNum.format(max / 1024 / 1024)+ " MB");

        for(int i = 0; i < 10; i++) {
            MatrixAssertion.assertDistancesWithSelfDataDouble(getRandomLargeMatrixDouble(i), null, null, execEnv, algorithm, -1, false);
            System.gc();
            MatrixSetManager.getInstance().clear();
            long free = Runtime.getRuntime().freeMemory();
            long used = total - free;
            System.out.println("Trial: # "+i);
            System.out.println("  free  => " + nfNum.format(free / 1024 / 1024) + " MB");
            System.out.println("  used  => " + nfNum.format((total - free) / 1024 / 1024) + " MB");
        }
    }
}
