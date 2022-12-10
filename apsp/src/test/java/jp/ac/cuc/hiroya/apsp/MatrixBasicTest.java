package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import jp.ac.cuc.hiroya.apsp.util.CSVParser;
import jp.ac.cuc.hiroya.apsp.util.InfinityConverter;
import org.junit.jupiter.api.Test;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;

public class MatrixBasicTest {
    @Test
    public void java側のDoubleのInf値はネイティブライブラリ側のInf値と一致する() {
        double inf = Infinity.DBL_INF;
        double[] expectedInf = ApspResolvers.DoubleResolver.getInfinity("omp");
        assertThat(inf, is(expectedInf[0]));
    }

    @Test
    public void java側のFloatのInf値はネイティブライブラリ側のInf値と一致する() {
        float inf = Infinity.FLT_INF;
        float[] expectedInf = ApspResolvers.FloatResolver.getInfinity("omp");
        assertThat(inf, is(expectedInf[0]));
    }

    @Test
    public void java側のIntのInf値はネイティブライブラリ側のInf値と一致する() {
        int inf = Infinity.INT_INF;
        int[] expectedInf = ApspResolvers.IntResolver.getInfinity("omp");
        assertThat(inf, is(expectedInf[0]));
    }

    @Test
    public void 簡易データの距離行列を読み込んだ結果が一致する() throws Exception {
        String filename = MatrixDummyFilenames.adjFilename;
        double[] expectedMatrix = {
                0, 1, 2, 0, 0,
                1, 0, 0, 1, 4,
                2, 0, 0, 3, 0,
                0, 1, 3, 0, 5,
                0, 4, 0, 5, 0,
        };
        double[] matrix = CSVParser.parseDoubleCSV(filename);//InfinityConverter.convert(, 0.0, ApspResolver.INF.DBL_INF);
        int expectedLength = (int) Math.pow(5, 2);
        assertThat(matrix.length, is(expectedLength));
        assertThat(matrix, is(expectedMatrix));
    }

    @Test
    public void 簡易データの距離行列をInf値を置き換えした結果が一致する() throws Exception {
        String filename = MatrixDummyFilenames.adjFilename;
        double inf = Infinity.DBL_INF;
        double[] expectedMatrix = {
                0, 1, 2, inf, inf,
                1, 0, inf, 1, 4,
                2, inf, 0, 3, inf,
                inf, 1, 3, 0, 5,
                inf, 4, inf, 5, 0,
        };
        double[] matrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
        int expectedLength = (int) Math.pow(5, 2);
        assertThat(matrix.length, is(expectedLength));
        assertThat(matrix, is(expectedMatrix));
    }
    /*
    @Test
    public void 本番データの距離行列が読み込める() throws Exception {
        String filename = MatrixRealFilenames.adjFilename;
        double[] matrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
        int expectedLength = (int) Math.pow(11083, 2);
        assertThat(matrix.length, is(expectedLength));
        assertThat(matrix[0], is(0.0));
        assertThat(matrix[matrix.length-1], is(0.0));
    }

    @Test
    public void 本番データの距離行列の対角要素はゼロである() throws Exception {
        String filename = MatrixRealFilenames.adjFilename;
        double[] adjMatrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
        int v = (int) Math.sqrt(adjMatrix.length);
        MatrixAssertion.assertDiagonalElementsAllZero(adjMatrix, v, false);
    }

    @Test
    public void 本番データの距離行列は対象行列である() throws Exception {
        String filename = MatrixRealFilenames.adjFilename;
        double[] adjMatrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
        int v = (int) Math.sqrt(adjMatrix.length);
        MatrixAssertion.assertSymmetricMatrix(adjMatrix, v, false);
    }
*/
}
