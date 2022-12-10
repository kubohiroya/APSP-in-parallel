package jp.ac.cuc.hiroya.apsp;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class MatrixDummyDataTest {

    static String adjFilename = MatrixDummyFilenames.adjFilename;
    static String distanceFilename = MatrixDummyFilenames.distanceFilename;
    static String successorFilename = MatrixDummyFilenames.successorFilename;

    @Test
    public void ダミーデータの整合性検証() throws Exception {
        MatrixAssertion.assertConsistencyOfProvidedCsvSet(adjFilename, distanceFilename, successorFilename, false);
    }

    @Test
    public void ダミーデータのFloydWarshall法での処理結果を外部ファイルで検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithProvidedData(adjFilename, distanceFilename, successorFilename, execEnv, algorithm, 16, false);
    }

    @Test
    public void ダミーデータのJohnson法での処理結果を外部ファイルで検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithProvidedData(adjFilename, distanceFilename, successorFilename, execEnv, algorithm, 16, false);
    }
}
