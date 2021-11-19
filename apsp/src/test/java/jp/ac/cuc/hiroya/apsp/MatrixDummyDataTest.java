package jp.ac.cuc.hiroya.apsp;

import org.junit.Test;

public class MatrixDummyDataTest {

    static String adjFilename = MatrixDummyFilenames.adjFilename;
    static String distanceFilename = MatrixDummyFilenames.distanceFilename;
    static String successorFilename = MatrixDummyFilenames.nodeFilename;

    @Test
    public void ダミーデータの整合性検証() throws Exception {
        MatrixAssertion.assertConsistencyOfProvidedCsvSet(adjFilename, distanceFilename, successorFilename, false);
    }

    @Test
    public void ダミーデータのFloydWarshall法での処理結果を外部ファイルで検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithProvidedData(adjFilename, distanceFilename, successorFilename, execEnv, algorithm, true);
    }

    @Test
    public void ダミーデータのJohnson法での処理結果を外部ファイルで検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithProvidedData(adjFilename, distanceFilename, successorFilename, execEnv, algorithm, false);
    }
}
