package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.ApspResult;

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

    Map<String,ApspResult<double[]>> cache = new HashMap();

    AbstractMatrixDetailedTest(String adjFilename, String distanceFilename, String nodeFilename)throws Exception{
        System.out.println("step 1: load csv " + new Date());
        this.adjMatrix = MatrixTestUtils.loadAdjMatrix(adjFilename);
        this.distanceMatrix = loadDistanceMatrix(distanceFilename);
        this.nodeMatrix = loadNodeMatrix(nodeFilename);
    }

    void assertCsvSet(boolean verbose) {
        System.out.println("step 2: calc successors " + new Date());
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
                                          boolean verbose) throws Exception {
        System.out.println("step 2: process " + new Date());
        ApspResult<double[]> result = getResult(execEnv, algorithm);
        int v = result.getNumVertex();
        double[] distances = result.getDistanceMatrix();
        int[] nodes = result.getSuccessorMatrix();

        System.out.println("step 3: assert distances " + new Date());
        if(verbose){
            for (int count = 0, i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (distances[index] != distanceMatrix[index]) {
                        System.out.println(i + "," + j + " actual:" + distances[index] + " expected:" + distanceMatrix[index]);
                        if(count++ > 10) {
                            System.out.println("...cancel step 4 assertion");
                            break;
                        }
                    }
                    //assertThat(distances[index], is(distanceMatrix[index]));
                }
            }
        }else{
            assertThat(distances, is(distanceMatrix));
        }

        System.out.println("step 4: assert nodes " + new Date());
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

        System.out.println("step 5: calc successors " + new Date());
        double[] calculatedDistances = generateDistanceMatrix(distances, nodes, false);

        System.out.println("step 6: assert calculated distance " + new Date());
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
                                 boolean verbose) throws Exception {
        System.out.println("step 2: process " + new Date());
        ApspResult<double[]> result = getResult(execEnv, algorithm);
        int v = result.getNumVertex();
        System.out.println("step 3: assert distances " + new Date());
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

