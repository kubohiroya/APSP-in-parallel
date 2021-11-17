package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.ApspResult;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.TimeZone;

import static jp.ac.cuc.hiroya.apsp.MatrixTestUtils.*;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

abstract class AbstractMatrixDetailedTest {

    double[] adjacencyMatrix;
    double[] distanceMatrix;
    int[] successorMatrix;

    Map<String,ApspResult<double[]>> cache = new HashMap<>();

    static TimeZone timeZoneJP = TimeZone.getTimeZone("Asia/Tokyo");
    static SimpleDateFormat sdf = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
    static{
        sdf.setTimeZone(timeZoneJP);
    }

    String getYYMMDDHHMM(){
        return sdf.format(new Date());
    }

    AbstractMatrixDetailedTest(String adjacencyFilename, String distanceFilename, String successorFilename)throws Exception{
        System.out.println(getYYMMDDHHMM()+" step 1: load csv ");
        System.out.println("   adj: " + adjacencyFilename);
        System.out.println("   dst: " + distanceFilename);
        System.out.println("   suc: " + successorFilename);
        this.adjacencyMatrix = MatrixTestUtils.loadAdjacencyMatrix(adjacencyFilename);
        this.distanceMatrix = loadDistanceMatrix(distanceFilename);
        this.successorMatrix = loadSuccessorMatrix(successorFilename);
    }

    void assertCsvSet(boolean verbose) {
        System.out.println(getYYMMDDHHMM()+" step 2: calculate distances with successor matrix");
        double[] calculatedDistances = generateDistanceMatrix(this.adjacencyMatrix, this.successorMatrix, verbose);
        System.out.println(getYYMMDDHHMM()+" step 3: assert calculated distance");
        assertThat(calculatedDistances, is(distanceMatrix));
    }

    ApspResult<double[]> getResult(String execEnv, String algorithm){
        String key = execEnv+":"+algorithm;
        ApspResult<double[]>result = cache.get(key);
        if(result == null){
            int v = (int) Math.sqrt(adjacencyMatrix.length);
            result = ApspResolvers.DoubleResolver.resolve(execEnv, algorithm,
                    adjacencyMatrix, v, 64);
            cache.put(key, result);
        }
        return result;
    }

    void assertAlgorithmWithTestData(String execEnv, String algorithm,
                                     boolean verbose) {
        System.out.println(getYYMMDDHHMM()+" step 2: resolve all-pairs-shortest-paths " + execEnv + "-" + algorithm);
        ApspResult<double[]> result = getResult(execEnv, algorithm);
        int v = result.getNumVertex();
        double[] distances = result.getDistanceMatrix();
        int[] successors = result.getSuccessorMatrix();

        System.out.println(getYYMMDDHHMM()+" step 3: assert distances with test data");
        if(verbose){
            for (int count = 0, i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (distances[index] != distanceMatrix[index]) {
                        System.out.println(i + "," + j + " actual:" + distances[index] + " expected:" + distanceMatrix[index]);
                        count++;
                    }
                    //assertThat(distances[index], is(distanceMatrix[index]));
                    if(count > 10) {
                        break;
                    }
                }
                if(count > 10) {
                    System.out.println("  ...cancel step 3 assertion");
                    break;
                }
            }
        }else{
            assertThat(distances, is(distanceMatrix));
        }

        System.out.println(getYYMMDDHHMM()+" step 4: assert successors with test data");
        if(false){
            // disable assertion of successors: there are more than one answers
            if(verbose){
                for (int count = 0, i = 0; i < v; i++) {
                    for (int j = 0; j < v; j++) {
                        int index = i * v + j;
                        if (successors[index] != successorMatrix[index]) {
                            System.out.println("i=" + i + ", j=" + j + "  successor= actual:" + successors[index] + " expected:" + successorMatrix[index]);
                            if(count++ > 10) {
                                System.out.println("...cancel step 4 assertion");
                                break;
                            }
                        }
                        assertThat(successors[index], is(successorMatrix[index])); // REMOVE
                    }
                }
            }else{
                assertThat(successors, is(successorMatrix)); // REMOVE
            }
        }else{
            System.out.println("  ignore step 4 assertion, various algorithm generated various successor matrix");
        }

        System.out.println(getYYMMDDHHMM()+" step 5: calculate distances with successor matrix");
        double[] calculatedDistances = generateDistanceMatrix(adjacencyMatrix, successors, false);

        System.out.println(getYYMMDDHHMM()+" step 6: assert calculated distance");
        if(verbose){
            for (int i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (calculatedDistances[index] != distanceMatrix[index]) {
                        System.out.println("("+i + "," + j + ") Successor "+
                                " actual:" + successors[index] + " expected:" + successorMatrix[index]);
                        if(calculatedDistances[index] != distanceMatrix[index]){
                            System.out.println("("+i + "," + j + ") DISTANCE actual:");
                            MatrixTestUtils.calculateDistance(i, j, v, distances, successors, true);
                            System.out.println("("+i + "," + j + "         expected:");
                            MatrixTestUtils.calculateDistance(i, j, v, distanceMatrix, successors, true);
                            System.out.println();
                        }
                    }
                    assertThat(calculatedDistances[index], is(distanceMatrix[index]));
                }
            }
        }else{
            assertThat(calculatedDistances, is(distanceMatrix));
        }
    }

    void assertAlgorithmWithSelfData(String execEnv, String algorithm,
                                     boolean verbose) {
        System.out.println(getYYMMDDHHMM()+" step 2: resolve all-pairs-shortest-paths " + execEnv + "-" + algorithm);
        ApspResult<double[]> result = getResult(execEnv, algorithm);
        int v = result.getNumVertex();

        System.out.println(getYYMMDDHHMM()+" step 3: assert calculated distance with self-generated data");
        double[] distances = result.getDistanceMatrix();
        int[] successors = result.getSuccessorMatrix();
        double[] calculatedDistances = generateDistanceMatrix(adjacencyMatrix, successors, verbose);

        if(verbose){
            for (int i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (calculatedDistances[index] != distances[index]) {
                        System.out.println("    "+i + "," + j + " actual:" + calculatedDistances[index] + " expected:" + distances[index]);
                        MatrixTestUtils.calculateDistance(i, j, v, distances, successors, true);
                    }
                    assertThat(calculatedDistances[index], is(distances[index]));
                }
            }
        }else{
            assertThat(calculatedDistances, is(distances));
        }
    }
}

