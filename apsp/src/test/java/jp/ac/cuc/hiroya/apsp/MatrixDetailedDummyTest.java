package jp.ac.cuc.hiroya.apsp;

import org.junit.Test;

public class MatrixDetailedDummyTest extends AbstractMatrixDetailedTest {

    static String adjFilename = MatrixTestFilenames.adjFilename;
    static String distanceFilename = MatrixTestFilenames.distanceFilename;
    static String nodeFilename = MatrixTestFilenames.nodeFilename;

    public MatrixDetailedDummyTest()throws Exception{
        super(adjFilename, distanceFilename, nodeFilename);
    }

    @Test
    public void 簡易データの整合性チェック() throws Exception {
        assertCsvSet(false);
    }

    @Test
    public void 簡易データをFloydWarshall法で処理結果チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        assertAlgorithmByTestData(execEnv, algorithm, false);
    }

    @Test
    public void 簡易データをJohnson法で処理結果チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        assertAlgorithmByTestData(execEnv, algorithm, false);
    }
}
