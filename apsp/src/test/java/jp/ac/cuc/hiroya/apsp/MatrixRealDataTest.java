package jp.ac.cuc.hiroya.apsp;

import org.junit.Test;

public class MatrixRealDataTest extends AbstractMatrixDetailedTest {

    static String adjFilename = MatrixRealFilenames.adjFilename;
    static String distanceFilename = MatrixRealFilenames.distanceFilename;
    static String nodeFilename = MatrixRealFilenames.nodeFilename;

    public MatrixRealDataTest()throws Exception{
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
        assertAlgorithmWithSelfData(execEnv, algorithm, false);
    }

    @Test
    public void 本番データをJohnson法で整合性チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        assertAlgorithmWithSelfData(execEnv, algorithm, false);
    }

    @Test
    public void 本番データをFloydWarshall法で処理結果チェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        assertAlgorithmWithTestData(execEnv, algorithm, true);
    }

    @Test
    public void 本番データをJohnson法で処理結果をチェック() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        assertAlgorithmWithTestData(execEnv, algorithm,
                true);
    }
}
