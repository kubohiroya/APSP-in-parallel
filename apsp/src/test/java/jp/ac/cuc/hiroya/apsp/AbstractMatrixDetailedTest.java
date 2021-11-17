package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.ApspResult;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import static jp.ac.cuc.hiroya.apsp.MatrixTestUtils.*;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

abstract class AbstractMatrixDetailedTest {

    double[] adjMatrix;
    double[] distanceMatrix;
    int[] nodeMatrix;

    Map<String,ApspResult<double[]>> cache = new HashMap<>();

    //日付をyyyyMMddの形で出力する
    SimpleDateFormat sdf = new SimpleDateFormat("yyyy/MM/dd hh:mm:ss");

    String getYYMMDDHHMM(Date date){
        return sdf.format(date.getTime());
    }

    AbstractMatrixDetailedTest(String adjFilename, String distanceFilename, String nodeFilename)throws Exception{
        System.out.println(getYYMMDDHHMM(new Date())+" step 1: load csv ");
        System.out.println("   adj: " + adjFilename);
        System.out.println("   dst: " + distanceFilename);
        System.out.println("   suc: " + nodeFilename);
        this.adjMatrix = MatrixTestUtils.loadAdjMatrix(adjFilename);
        this.distanceMatrix = loadDistanceMatrix(distanceFilename);
        this.nodeMatrix = loadNodeMatrix(nodeFilename);
    }

    void assertCsvSet(boolean verbose) {
        System.out.println(getYYMMDDHHMM(new Date())+" step 2: calc successors");
        double[] calculatedDistances = generateDistanceMatrix(this.distanceMatrix, this.nodeMatrix, verbose);
        System.out.println("step 3: assert calculated distance " + new Date());
        assertThat(calculatedDistances, is(distanceMatrix));
    }

    ApspResult<double[]> getResult(String execEnv, String algorithm){
        String key = execEnv+":"+algorithm;
        ApspResult<double[]>result = cache.get(key);
        if(result == null){
            int v = (int) Math.sqrt(adjMatrix.length);
            result = ApspResolvers.DoubleResolver.resolve(execEnv, algorithm,
                    adjMatrix, v, 64);
            cache.put(key, result);
        }
        return result;
    }

    void assertAlgorithmByTestData(String execEnv, String algorithm,
                                          boolean verbose) {
        System.out.println(getYYMMDDHHMM(new Date())+" step 2: resolve all-pairs-shortest-paths " + execEnv + "-" + algorithm);
        ApspResult<double[]> result = getResult(execEnv, algorithm);
        int v = result.getNumVertex();
        double[] distances = result.getDistanceMatrix();
        int[] nodes = result.getSuccessorMatrix();

        System.out.println(getYYMMDDHHMM(new Date())+" step 3: assert distances");
        if(verbose){
            for (int count = 0, i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (distances[index] != distanceMatrix[index]) {
                        System.out.println(i + "," + j + " actual:" + distances[index] + " expected:" + distanceMatrix[index]);
                    }
                    count++;
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

        System.out.println(getYYMMDDHHMM(new Date())+" step 4: assert nodes");
        if(false){
            // disable assertion of nodes: there are more than one answers
            if(verbose){
                for (int count = 0, i = 0; i < v; i++) {
                    for (int j = 0; j < v; j++) {
                        int index = i * v + j;
                        if (nodes[index] != nodeMatrix[index]) {
                            System.out.println("i=" + i + ", j=" + j + "  node= actual:" + nodes[index] + " expected:" + nodeMatrix[index]);
                            if(count++ > 10) {
                                System.out.println("...cancel step 4 assertion");
                                break;
                            }
                        }
                        assertThat(nodes[index], is(nodeMatrix[index])); // REMOVE
                    }
                }
            }else{
                assertThat(nodes, is(nodeMatrix)); // REMOVE
            }
        }

        System.out.println(getYYMMDDHHMM(new Date())+" step 5: calc successors");
        double[] calculatedDistances = generateDistanceMatrix(distances, nodes, false);

        System.out.println(getYYMMDDHHMM(new Date())+" step 6: assert calculated distance");
        if(verbose){
            for (int i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (calculatedDistances[index] != distanceMatrix[index]) {
                        System.out.println("("+i + "," + j + ") NODE"+
                                " actual:" + nodes[index] + " expected:" + nodeMatrix[index]);
                        if(calculatedDistances[index] != distanceMatrix[index]){
                            System.out.println("("+i + "," + j + ") DISTANCE actual:");
                            MatrixTestUtils.calculateDistance(i, j, v, distances, nodes, true);
                            System.out.println("("+i + "," + j + "         expected:");
                            MatrixTestUtils.calculateDistance(i, j, v, distanceMatrix, nodes, true);
                            System.out.println();
                        }
                    }
                    // assertThat(nodes[index], is(nodeMatrix[index]));
                    assertThat(calculatedDistances[index], is(distanceMatrix[index]));
                }
            }
        }else{
            assertThat(calculatedDistances, is(distanceMatrix));
        }
    }

    void assertAlgorithmByItself(String execEnv, String algorithm,
                                 boolean verbose) {
        System.out.println(getYYMMDDHHMM(new Date())+" step 2: process");
        ApspResult<double[]> result = getResult(execEnv, algorithm);
        int v = result.getNumVertex();
        System.out.println(getYYMMDDHHMM(new Date())+" step 3: assert distances");
        double[] distances = result.getDistanceMatrix();
        int[] nodes = result.getSuccessorMatrix();
        double[] calculatedDistances = generateDistanceMatrix(distances, nodes, verbose);

        // assertThat(calculatedDistances, is(distances)); // FIXME
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                int index = i * v + j;
                if (calculatedDistances[index] != distances[index]) {
                    System.out.println(i + "," + j + " actual:" + calculatedDistances[index] + " expected:" + distances[index]);
                    MatrixTestUtils.calculateDistance(i, j, v, distances, nodes, true);
                }
                assertThat(calculatedDistances[index], is(distances[index]));
            }
        }
    }
}

