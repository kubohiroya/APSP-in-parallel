package jp.ac.cuc.hiroya.apsp;

import org.junit.Test;

public class MatrixDetailedTestWithRealData extends AbstractMatrixDetailedTest {

    static String adjFilename = MatrixRealFilenames.adjFilename;
    static String distanceFilename = MatrixRealFilenames.distanceFilename;
    static String nodeFilename = MatrixRealFilenames.nodeFilename;

    public MatrixDetailedTestWithRealData()throws Exception{
        super(adjFilename, distanceFilename, nodeFilename);
    }

    @Test
    public void 本番データの整合性チェック() throws Exception {
        assertCsvSet(false);
    }

    @Test
    public void 本番データをFloydWarshall法で整合性チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        assertAlgorithmByItself(execEnv, algorithm, false);
    }

    @Test
    public void 本番データをJohnson法で整合性チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        assertAlgorithmByItself(execEnv, algorithm, false);
    }

    @Test
    public void 本番データをFloydWarshall法で処理結果チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        assertAlgorithmByTestData(execEnv, algorithm, true);
    }

    @Test
    public void 本番データをJohnson法で処理結果をチェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        assertAlgorithmByTestData(execEnv, algorithm,
                true);
    }
}
