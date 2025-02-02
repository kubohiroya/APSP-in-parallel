package jp.ac.cuc.hiroya.apsp;

import org.junit.jupiter.api.Test;
public class MatrixRealDataTest {

    static String adjFilename = MatrixRealFilenames.adjFilename;
    static String distanceFilename = MatrixRealFilenames.distanceFilename;
    static String successorFilename = MatrixRealFilenames.successorFilename;

    @Test
    public void 本番データの整合性検証() throws Exception {
        MatrixAssertion.assertConsistencyOfProvidedCsvSet(adjFilename, distanceFilename, successorFilename,false);
    }

    @Test
    public void 本番データのFloydWarshall法での処理結果を外部ファイルで検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithProvidedData(adjFilename, distanceFilename, successorFilename, execEnv, algorithm, 32, false);
    }

    @Test
    public void 本番データのBlockedFloydWarshall法での処理結果を外部ファイルで検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithProvidedData(adjFilename, distanceFilename, successorFilename, execEnv, algorithm, 32, false);
    }

    @Test
    public void 本番データのJohnson法での処理結果を外部ファイルで検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithProvidedData(adjFilename, distanceFilename, successorFilename, execEnv, algorithm, 32, false);
    }

}
