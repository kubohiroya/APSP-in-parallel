package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResult;
import jp.ac.cuc.hiroya.apsp.util.CSVOutput;

import java.io.IOException;

import static jp.ac.cuc.hiroya.apsp.DateTimeUtil.getYYMMDDHHMM;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

public class MatrixAssertion {

    static void assertConsistencyOfProvidedCsvSet(String adjacencyFilename,
                                                  String distanceFilename,
                                                  String successorFilename, boolean verbose) throws IOException{
        System.out.println(getYYMMDDHHMM()+" step A: calculate distances with successor matrix");
        MatrixSet.Double set = MatrixSetManager.getInstance().getMatrixSetDouble(adjacencyFilename, distanceFilename, successorFilename);
        double[] calculatedDistances = MatrixUtil.calculateDistanceMatrix(set.adjacencyMatrix, set.successorMatrix, verbose);
        System.out.println(getYYMMDDHHMM()+" step B: assert calculated distance");
        assertThat(calculatedDistances, is(set.distanceMatrix));
    }

    static void assertDistancesWithSelfData(String adjacencyFilename,
                                            String distanceFilename,
                                            String successorFilename,
                                            String execEnv, String algorithm,
                                            boolean verbose) throws IOException {
        MatrixSet.Double set = MatrixSetManager.getInstance().getMatrixSetDouble(adjacencyFilename, distanceFilename, successorFilename);
        ApspResult<double[]> result = MatrixSetManager.getInstance().getApspResultDouble(adjacencyFilename, execEnv, algorithm);
        System.out.println(getYYMMDDHHMM() + " step A: assert calculated distance with self-generated data");
        double[] distances = result.getDistanceMatrix();
        int[] successors = result.getSuccessorMatrix();
        double[] calculatedDistances = MatrixUtil.calculateDistanceMatrix(set.getAdjacencyMatrix(), successors, false);
        int v = set.getNumVertex();
        if(verbose){
            System.out.println("successor:");
            CSVOutput.print(successors, v);
            System.out.println("distances:");
            CSVOutput.print(distances, v);
            System.out.println("calculated distances:");
            CSVOutput.print(calculatedDistances, v);
        }
        assertThat(v*v, is(calculatedDistances.length));
        if (verbose) {
            for (int i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (calculatedDistances[index] != distances[index]) {
                        System.out.println("*** " + i + "," + j + " actual:" + calculatedDistances[index] + " expected:" + distances[index]);
                        MatrixUtil.calculateDistance(i, j, v, set.getAdjacencyMatrix(), successors, true);
                        assertEquals(calculatedDistances[index], distances[index], 0.0001);
                    }
                }
            }
        } else {
            assertThat(calculatedDistances, is(distances));
        }
    }

    static void assertDistancesWithProvidedData(String adjacencyFilename,
                                                String distanceFilename,
                                                String successorFilename,
                                                String execEnv, String algorithm,
                                                boolean verbose) throws IOException{
        MatrixSet.Double set = MatrixSetManager.getInstance().getMatrixSetDouble(adjacencyFilename, distanceFilename, successorFilename);
        ApspResult<double[]> result = MatrixSetManager.getInstance().getApspResultDouble(adjacencyFilename, execEnv, algorithm);
        System.out.println(getYYMMDDHHMM()+" step A: assert distances with test data");
        double[] distances = result.getDistanceMatrix();
        int v = set.getNumVertex();
        if(verbose){
            System.out.println("successor:");
            CSVOutput.print(result.getSuccessorMatrix(), v);
            System.out.println("distances:");
            CSVOutput.print(result.getDistanceMatrix(), v);
            System.out.println("provided distances:");
            CSVOutput.print(set.getDistanceMatrix(), v);

            for (int count = 0, i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (distances[index] != set.getDistanceMatrix()[index]) {
                        System.out.println(i + "," + j + " actual:" + distances[index] + " expected:" + set.getDistanceMatrix()[index]);
                        assertThat(distances[index], is(set.getDistanceMatrix()[index]));
                        count++;
                    }
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
            assertThat(distances, is(set.getDistanceMatrix()));
        }
    }

}

