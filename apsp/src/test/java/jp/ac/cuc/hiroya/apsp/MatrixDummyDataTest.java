package jp.ac.cuc.hiroya.apsp;

import org.junit.Test;

public class MatrixDummyDataTest extends AbstractMatrixDetailedTest {

    static String adjFilename = MatrixDummyFilenames.adjFilename;
    static String distanceFilename = MatrixDummyFilenames.distanceFilename;
    static String nodeFilename = MatrixDummyFilenames.nodeFilename;

    public MatrixDummyDataTest()throws Exception{
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
        assertAlgorithmWithTestData(execEnv, algorithm, false);
    }

    @Test
    public void 簡易データをJohnson法で処理結果チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        assertAlgorithmWithTestData(execEnv, algorithm, false);
    }
}
