package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import org.junit.Test;

import java.io.IOException;

public class MatrixRandomDataTest {

    String getAdjName() throws IOException {
        long seed = 0;
        int n = 16;
        double p = 0.25;
        double min = 1.0;
        double max = 100.0;
        double inf = Infinity.DBL_INF;
        return "random:"+seed+","+n+","+p+","+min+","+max+",inf";
    }

    @Test
    public void ランダムデータのFloydWarshall法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "f";
        MatrixAssertion.assertDistancesWithSelfData(getAdjName(), null, null, execEnv, algorithm, true);
    }

    @Test
    public void ランダムデータのJohnson法での処理結果の整合性を自己検証() throws Exception {
        String execEnv = "omp-ispc";
        String algorithm = "j";
        MatrixAssertion.assertDistancesWithSelfData(getAdjName(), null, null, execEnv, algorithm, true);
    }

}
