package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.ApspResult;
import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import jp.ac.cuc.hiroya.apsp.util.CSVParser;
import jp.ac.cuc.hiroya.apsp.util.InfinityConverter;
import jp.ac.cuc.hiroya.apsp.util.PostcedessorNormalizer;
import org.junit.Test;

import java.util.Date;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class AppTest {

    interface TinyMatrix{
        String adjFilename = "../InputMatrix.csv";
        String distanceFilename = "../inputMatrix(distance).csv";
        String nodeFilename = "../inputMatrix(node).csv";
    }
    interface HugeMatrix{
        String adjFilename = "../DDMATAgriculture.csv";
        String distanceFilename = "../DDMATAgriculture(distance).csv";
        String nodeFilename = "../DDMATAgriculture(node).csv";
    }

    @Test
    public void java側のinfinity値はネイティブライブラリ側のinfinityと一致する() throws Exception {
        double inf = Infinity.DBL_INF;
        double[] expectedInf = ApspResolvers.DoubleResolver.getInfinity("omp");
        assertThat(inf, is(expectedInf[0]));
    }

    @Test
    public void テスト用距離行列がdouble配列としてparseできる() throws Exception {
        String filename = TinyMatrix.adjFilename;
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
    public void テスト用距離行列がparseできる() throws Exception {
        String filename = TinyMatrix.adjFilename;
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

    @Test
    public void 本番用距離行列がparseできる() throws Exception {
        String filename = HugeMatrix.adjFilename;
        double[] matrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
        int expectedLength = (int) Math.pow(11083, 2);
        assertThat(matrix.length, is(expectedLength));
        assertThat(matrix[0], is(0.0));
        assertThat(matrix[matrix.length-1], is(0.0));
    }

    @Test
    public void 本番用距離行列の対角要素はゼロ() throws Exception {
        String filename = HugeMatrix.adjFilename;
        double inf = Infinity.DBL_INF;
        double[] adjMatrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
        int v = (int) Math.sqrt(adjMatrix.length);
        for(int i=0; i<v; i++) {
            if(adjMatrix[i * v + i] != 0.0){
                System.out.println(i+"="+adjMatrix[i * v + i]);;
            }
        }
        for(int i=0; i<v; i++) {
            assertThat(adjMatrix[i * v + i], is(0.0));
        }
    }

    static double[] generateDistanceMatrix(double[] distances, int[] nodes,
                             boolean verbose)throws Exception{
        int v = (int) Math.sqrt(distances.length);
        double[] ret = new double[v * v];
        for(int i = 0; i < v; i++){
            for(int j = 0; j < v; j++){
                int index = i * v + j;
                if(verbose)
                    System.out.println(i+"発 => "+j+"行\t"+
                            distances[index]);

                double distance = 0.0;
                int from = i;
                int to = j;
                while(true){
                    int next = nodes[from * v + to];
                    if(from == next){
                        distance = 0.0;
                        break;
                    }
                    distance += distances[from * v + next];
                    if(next == to){
                        if(verbose) System.out.println("   "+from+"発 => "+to+"行\t最終合計距離="+distance);
                        break;
                    }
                    if(verbose) System.out.println("   "+from+"発 => "+next+"経由 (最終目的地"+to+")\t合計距離="+distance);
                    from = next;
                }
                ret[i*v + j] = distance;
            }
        }
        return ret;
    }

    static void 総合チェックDouble(String adjFilename, String distanceFilename, String nodeFilename, String execEnv, String algorithm,
                             boolean checkCalcDistanceAlgorithm,
                             boolean checkCalcPostdecessorAlgorithm,
                             boolean verbose)throws Exception{
        if(verbose) System.out.println("step 0 "+ new Date());
        double[] adjMatrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(adjFilename), 0.0, Infinity.DBL_INF);
        if(verbose) System.out.println("step 1 "+ new Date());
        double[] distanceMatrix = InfinityConverter.convert(CSVParser.parseDoubleCSV(distanceFilename), 0.0, Infinity.DBL_INF);
        if(verbose) System.out.println("step 2 "+ new Date());
        int[] nodeMatrix = PostcedessorNormalizer.normalize(CSVParser.parseIntCSV(nodeFilename));
        //int[] nodeMatrix = CSVParser.parseIntCSV(nodeFilename);
        if(verbose) System.out.println("step 3 "+ new Date());
        int v = (int) Math.sqrt(adjMatrix.length);
        assertThat(v * v, is(distanceMatrix.length));
        assertThat(v * v, is(nodeMatrix.length));
        System.out.println("step 4 "+ new Date());
        ApspResult<double[]> result = ApspResolvers.DoubleResolver.resolve(execEnv, algorithm,
                adjMatrix, v, 64);
        System.out.println("step 5 "+ new Date());
        assertThat(v, is(result.getNumVertex()));
        assertThat(v * v, is(result.getOutput().length));
        assertThat(v * v, is(result.getPostdecessors().length));

        double[] distances = checkCalcDistanceAlgorithm? result.getOutput() : distanceMatrix;
        int[] nodes = checkCalcPostdecessorAlgorithm? result.getPostdecessors() : nodeMatrix;

        double[] calculatedDistances = generateDistanceMatrix(distances, nodes, true);
        System.out.println("step 6 "+ new Date());

        for(int i = 0; i < v; i++){
            for(int j = 0; j < v; j++){
                int index = i * v + j;
                if(verbose)
                    System.out.println(i+"発 => "+j+"行\t"+
                            result.getOutput()[index]+"\t"+
                            distanceMatrix[index]+"\t"+
                            // calculatedDistances[index]+"\t - "+
                            adjMatrix[index]+"\t - "+
                            nodes[index]
                    );
                assertThat(result.getOutput()[index], is(distanceMatrix[index]));
                //assertThat(result.getPostdecessors()[index], is(nodeMatrix[index]));
                assertThat(calculatedDistances[index], is(distances[index]));
            }
        }
    }

    @Test
    public void 簡易総合チェック1() throws Exception {
        String adjFilename = TinyMatrix.adjFilename;
        String distanceFilename = TinyMatrix.distanceFilename;
        String nodeFilename = TinyMatrix.nodeFilename;
        String execEnv = "omp-ispc";
        String algorithm = "f";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                false, false, false);
    };

    @Test
    public void 簡易総合チェック2() throws Exception {
        String adjFilename = TinyMatrix.adjFilename;
        String distanceFilename = TinyMatrix.distanceFilename;
        String nodeFilename = TinyMatrix.nodeFilename;
        String execEnv = "omp-ispc";
        String algorithm = "j";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                false, false,
                false);
    }

    @Test
    public void 簡易総合チェック3() throws Exception {
        String adjFilename = TinyMatrix.adjFilename;
        String distanceFilename = TinyMatrix.distanceFilename;
        String nodeFilename = TinyMatrix.nodeFilename;
        String execEnv = "omp-ispc";
        String algorithm = "f";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                true, true,
                false);
    };

    @Test
    public void 簡易総合チェック4() throws Exception {
        String adjFilename = TinyMatrix.adjFilename;
        String distanceFilename = TinyMatrix.distanceFilename;
        String nodeFilename = TinyMatrix.nodeFilename;
        String execEnv = "omp-ispc";
        String algorithm = "j";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                true, true,
                false);
    };

    @Test
    public void 総合チェック1() throws Exception {
        String adjFilename = HugeMatrix.adjFilename;
        String distanceFilename = HugeMatrix.distanceFilename;
        String nodeFilename = HugeMatrix.nodeFilename;
        String execEnv = "omp";
        String algorithm = "f";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                false, false,
                true);
    };

    @Test
    public void 総合チェック2() throws Exception {
        String adjFilename = HugeMatrix.adjFilename;
        String distanceFilename = HugeMatrix.distanceFilename;
        String nodeFilename = HugeMatrix.nodeFilename;
        String execEnv = "omp-ispc";
        String algorithm = "j";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                false, false, false);
    };

    @Test
    public void 総合チェック3() throws Exception {
        String adjFilename = HugeMatrix.adjFilename;
        String distanceFilename = HugeMatrix.distanceFilename;
        String nodeFilename = HugeMatrix.nodeFilename;
        String execEnv = "omp";
        String algorithm = "f";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                true, false,
                false);
    };

    @Test
    public void 総合チェック4() throws Exception {
        String adjFilename = HugeMatrix.adjFilename;
        String distanceFilename = HugeMatrix.distanceFilename;
        String nodeFilename = HugeMatrix.nodeFilename;
        String execEnv = "omp-ispc";
        String algorithm = "j";
        総合チェックDouble(adjFilename, distanceFilename, nodeFilename, execEnv, algorithm,
                true, true,
                true
        );
    };

}
