package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResult;
import jp.ac.cuc.hiroya.apsp.util.CSVOutput;

import java.io.IOException;

import static jp.ac.cuc.hiroya.apsp.util.ColorSeq.*;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.*;

public class MatrixAssertion {

    static void assertDiagonalElementsAllZero(double[] matrix, int v, boolean verbose){
        try{
            for(int i = 0; i < v; i++) {
                assertThat(matrix[i * v + i], is(0.0));
            }
        }catch(Error err){
            if(verbose)
                CSVOutput.print(matrix, (i, j, value) -> i == j && Math.abs(matrix[i * v + j]) > 0.0001, _red, v);
            throw err;
        }
    }

    static void assertDiagonalElementsAllZero(int[] matrix, int v, boolean verbose){
        try{
            for(int i = 0; i < v; i++) {
                assertThat(matrix[i * v + i], is(0));
            }
        }catch(Error err){
            if(verbose)
                CSVOutput.print(matrix, (i, j, value) -> i == j && Math.abs(matrix[i * v + j]) > 0.0001, _red, v);
            throw err;
        }
    }

    static void assertDiagonalElementsSequential(int[] matrix, int v, boolean verbose){
        try{
            for(int i = 0; i < v; i++) {
                assertThat(matrix[i * v + i], is(i));
            }
        }catch(Error err){
            if(verbose)
                CSVOutput.print(matrix, (i, j, value) -> i == j && Math.abs(matrix[i * v + j]) > 0.0001, _red, v);
            throw err;
        }
    }

    static void assertSymmetricMatrix(double[] matrix, int v, boolean verbose){
        try{
            for(int i = 0; i < v; i++) {
                for(int j = i; j < v; j++) {
                    if(Math.abs(matrix[i * v + j] - matrix[j * v + i]) > 0.0001){
                        System.err.println("対称行列ERROR:("+i+","+j+") != ("+j+","+i+")\tactual:"+matrix[i * v + j]+"\texpected:"+matrix[j * v + i]);
                        assertEquals(matrix[i * v + j], matrix[j * v + i], 0.0001);
                    }
                }
            }
        }catch(Error err){
            if(verbose)
                CSVOutput.print(matrix, (i, j, value) -> Math.abs(matrix[i * v + j] - matrix[j * v + i]) > 0.0001, _red, v);
            throw err;
        }
    }

    static void assertSymmetricMatrix(int[] matrix, int v, boolean verbose){
        if(verbose){
            CSVOutput.print(matrix, (i, j, value) -> Math.abs(matrix[i * v + j] - matrix[j * v + i]) > 0.0001, _red, v);
        }
        try{
            for(int i = 0; i < v; i++) {
                for(int j = i; j < v; j++) {
                    if(Math.abs(matrix[i * v + j] - matrix[j * v + i]) > 0.0001){
                        if(verbose) System.err.println("対称行列:("+i+","+j+") != ("+j+","+i+")\tactual:"+matrix[i * v + j]+"\texpected:"+matrix[j * v + i]);
                        assertEquals(matrix[i * v + j], matrix[j * v + i], 0.0001);
                    }
                }
            }
        }catch(Error ex){
            //CSVOutput.print(matrix, (i, j, value) -> Math.abs(matrix[i * v + j] - matrix[j * v + i]) > 0.0001, _red, v);
            throw ex;
        }
    }

    static void assertConsistencyOfProvidedCsvSet(String adjacencyFilename,
                                                  String distanceFilename,
                                                  String successorFilename, boolean verbose) throws IOException{
        if(verbose) {
            System.out.println(grey + DateTimeUtil.getYYMMDDHHMM() + " step A: calculate distances with successor matrix" + end);
        }
        MatrixSet.Double set = MatrixSetManager.getInstance().getMatrixSetDouble(adjacencyFilename, distanceFilename, successorFilename, verbose);
        if(verbose){
            System.out.println(grey+DateTimeUtil.getYYMMDDHHMM()+" step B: assertion"+end);
        }
        int v = set.getNumVertex();
        System.out.println(grey+DateTimeUtil.getYYMMDDHHMM()+" step B-1: adj matrix"+end);
        assertDiagonalElementsAllZero(set.getAdjacencyMatrix(), v, verbose);
        assertSymmetricMatrix(set.getAdjacencyMatrix(), v, verbose);
        System.out.println(grey+DateTimeUtil.getYYMMDDHHMM()+" step B-2: distance matrix"+end);
        assertDiagonalElementsAllZero(set.getDistanceMatrix(), v, verbose);
        assertSymmetricMatrix(set.getDistanceMatrix(), v, verbose);
        System.out.println(grey+DateTimeUtil.getYYMMDDHHMM()+" step B-3: calculated distance matrix"+end);
        double[] calculatedDistances = MatrixUtil.calculateDistanceMatrix(set.adjacencyMatrix, set.successorMatrix, verbose);
        assertDiagonalElementsAllZero(calculatedDistances, v, verbose);
        assertSymmetricMatrix(calculatedDistances, v, verbose);
        assertThat(calculatedDistances, is(set.getDistanceMatrix()));
    }

    static void assertDistancesWithSelfDataDouble(String adjacencyFilename,
                                                  String distanceFilename,
                                                  String successorFilename,
                                                  String execEnv, String algorithm,
                                                  int blocks,
                                                  boolean verbose) throws IOException {
        MatrixSet.Double set = MatrixSetManager.getInstance().getMatrixSetDouble(adjacencyFilename, distanceFilename, successorFilename, verbose);

        assertDiagonalElementsAllZero(set.getAdjacencyMatrix(), set.getNumVertex(), verbose);
        assertSymmetricMatrix(set.getAdjacencyMatrix(), set.getNumVertex(), verbose);

        ApspResult<double[]> result = MatrixSetManager.getInstance().getApspResultDouble(adjacencyFilename, execEnv, algorithm, blocks, verbose);
        if(verbose) {
            System.out.println(grey + DateTimeUtil.getYYMMDDHHMM() + " step A: assert calculated distance with self-generated data" + end);
        }
        double[] distances = result.getDistanceMatrix();
        int[] successors = result.getSuccessorMatrix();
        int v = set.getNumVertex();

        if(verbose){
            System.out.println(purple+"adjacency:"+end);
            CSVOutput.print(set.getAdjacencyMatrix(), v);
            System.out.println(purple+"successor:"+end);
            CSVOutput.print(successors, v);
            System.out.println(purple+"distances:"+end);
            CSVOutput.print(distances, v);
        }
        if(verbose) System.out.println(yellow+"check distances:"+end);
        assertDiagonalElementsAllZero(distances, v, verbose);
        assertSymmetricMatrix(distances, v, verbose);
        if(verbose) System.out.println(yellow+"check successors:"+end);
        assertDiagonalElementsSequential(successors, v, verbose);
        if(verbose) System.out.println(yellow+"check calculated distances:"+end);
        double[] calculatedDistances = MatrixUtil.calculateDistanceMatrix(set.getAdjacencyMatrix(), successors, verbose);
        assertDiagonalElementsAllZero(calculatedDistances, v, verbose);
        assertSymmetricMatrix(calculatedDistances, v, verbose);
        if (verbose) {
            for (int i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (Math.abs(calculatedDistances[index] - distances[index]) > 0.0001) {
                        System.err.println("*** " + i + "," + j + " actual:" + calculatedDistances[index] + " expected:" + distances[index]);
                        MatrixUtil.calculateDistance(i, j, v, set.getAdjacencyMatrix(), successors, true);
                        assertEquals(calculatedDistances[index], distances[index], 0.0001);
                    }
                }
            }
        } else {
            assertArrayEquals(calculatedDistances, distances, 0.0001);
        }
    }

    static void assertDistancesWithSelfDataInt(String adjacencyFilename,
                                               String distanceFilename,
                                               String successorFilename,
                                               String execEnv, String algorithm,
                                               int b,
                                               boolean verbose) throws IOException {
        MatrixSet.Int set = MatrixSetManager.getInstance().getMatrixSetInt(adjacencyFilename, distanceFilename, successorFilename, verbose);

        assertDiagonalElementsAllZero(set.getAdjacencyMatrix(), set.getNumVertex(), verbose);
        assertSymmetricMatrix(set.getAdjacencyMatrix(), set.getNumVertex(), verbose);

        ApspResult<int[]> result = MatrixSetManager.getInstance().getApspResultInt(adjacencyFilename, execEnv, algorithm, b, verbose);
        if(verbose) {
            System.out.println(grey + DateTimeUtil.getYYMMDDHHMM() + " step A: assert calculated distance with self-generated data" + end);
        }
        int[] distances = result.getDistanceMatrix();
        int[] successors = result.getSuccessorMatrix();
        int v = set.getNumVertex();

        if(verbose){
            System.out.println(purple+"adjacency:"+end);
            CSVOutput.print(set.getAdjacencyMatrix(), v);
            System.out.println(purple+"successor:"+end);
            CSVOutput.print(successors, v);
            System.out.println(yellow+"distances:"+end);
            CSVOutput.print(distances, v);
        }
        System.out.println(yellow+"check distances:"+end);
        assertDiagonalElementsAllZero(distances, v, verbose);
        assertSymmetricMatrix(distances, v, verbose);
        System.out.println(yellow+"check successors:"+end);
        assertDiagonalElementsSequential(successors, v, verbose);
        System.out.println(yellow+"check calculated distances:"+end);
        int[] calculatedDistances = MatrixUtil.calculateDistanceMatrix(set.getAdjacencyMatrix(), successors, verbose);
        assertDiagonalElementsAllZero(calculatedDistances, v, verbose);
        assertSymmetricMatrix(calculatedDistances, v, verbose);
        if (verbose) {
            for (int i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (Math.abs(calculatedDistances[index] - distances[index]) > 0.0001) {
                        System.err.println("*** " + i + "," + j + " actual:" + calculatedDistances[index] + " expected:" + distances[index]);
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
                                                int b,
                                                boolean verbose) throws IOException{
        MatrixSet.Double set = MatrixSetManager.getInstance().getMatrixSetDouble(adjacencyFilename, distanceFilename, successorFilename, verbose);
        ApspResult<double[]> result = MatrixSetManager.getInstance().getApspResultDouble(adjacencyFilename, execEnv, algorithm, b, verbose);
        if(verbose) {
            System.out.println(grey + DateTimeUtil.getYYMMDDHHMM() + " step A: assert distances with test data" + end);
        }
        double[] distances = result.getDistanceMatrix();
        int v = set.getNumVertex();

        assertDiagonalElementsAllZero(distances, v, verbose);
        //assertSymmetricMatrix(distances, v, verbose);
        assertDiagonalElementsSequential(result.getSuccessorMatrix(), v, verbose);

        if(true){//verbose
            /*
            System.out.println(purple+"adj:"+end);
            CSVOutput.print(set.getAdjacencyMatrix(), v);
            System.out.println(purple+"successor:"+end);
            CSVOutput.print(result.getSuccessorMatrix(), v);
            System.out.println(yellow+"distances:"+end);
            CSVOutput.print(result.getDistanceMatrix(), v);
            System.out.println(yellow+"provided distances:"+end);
            CSVOutput.print(set.getDistanceMatrix(), v);
             */
            for (int count = 0, i = 0; i < v; i++) {
                for (int j = 0; j < v; j++) {
                    int index = i * v + j;
                    if (Math.abs(distances[index] - set.getDistanceMatrix()[index]) > 0.00001) {
                        System.err.println("*** "+i + "," + j + " actual:" + distances[index] + " expected:" + set.getDistanceMatrix()[index]);
                        assertThat(distances[index], is(set.getDistanceMatrix()[index]));
                        count++;
                    }
                    if(count > 10) {
                        break;
                    }
                }
                if(count > 10) {
                    System.err.println("  ...cancel step 3 assertion");
                    break;
                }
            }
        }else{
            assertArrayEquals(distances, set.getDistanceMatrix(), 0.0001);
        }
    }

    static void assertDistancesBetweenAlgorithmsDouble(String adjacencyFilename,
                                                       String distanceFilename,
                                                       String successorFilename,
                                                       String[][] env,
                                                       int[] blocks,
                                                       boolean verbose) throws IOException{
        MatrixSet.Double set = MatrixSetManager.getInstance().getMatrixSetDouble(adjacencyFilename, distanceFilename, successorFilename, verbose);

        String execEnv0 = env[0][0];
        String algorithm0 = env[0][1];
        ApspResult<double[]> result0 = MatrixSetManager.getInstance().getApspResultDouble(adjacencyFilename, execEnv0, algorithm0, blocks[0], verbose);
        double[] distances0 = result0.getDistanceMatrix();
        int v = set.getNumVertex();

        for(int a = 1; a < env.length;  a++) {
            String execEnv1 = env[a][0];
            String algorithm1 = env[a][1];
            ApspResult<double[]> result1 = MatrixSetManager.getInstance().getApspResultDouble(adjacencyFilename, execEnv1, algorithm1, blocks[a], verbose);
            double[] distances1 = result1.getDistanceMatrix();
            try{
                assertArrayEquals(distances0, distances1, 0);
            }catch(Error err){
                // CSVOutput.printDiff(distances0, distances1, grey, red, green, v);
                for (int count = 1, i = 0; i < v; i++) {
                    for (int j = 0; j < v; j++) {
                        int index = i * v + j;
                        if (Math.abs(distances0[index] - distances1[index]) > 0.0001) {
                            System.err.println(count +": "+ i + "," + j + " actual:" + distances0[index] + " expected:" + distances1[index]);
                            assertEquals(distances0[index], distances1[index], 0.0001);
                            count++;
                        }
                        if(count >= 1000) {
                            // break;
                        }
                    }
                    if(count >= 1000) {
                        System.err.println("  ...cancel assertion");
                        break;
                    }
                }
            }
        }
    }

    static void assertDistancesBetweenAlgorithmsInt(String adjacencyFilename,
                                                    String distanceFilename,
                                                    String successorFilename,
                                                    String[][] env,
                                                    int[] blocks,
                                                    boolean verbose) throws IOException{
        MatrixSet.Int set = MatrixSetManager.getInstance().getMatrixSetInt(adjacencyFilename, distanceFilename, successorFilename, verbose);

        String execEnv0 = env[0][0];
        String algorithm0 = env[0][1];
        ApspResult<int[]> result0 = MatrixSetManager.getInstance().getApspResultInt(adjacencyFilename, execEnv0, algorithm0, blocks[0], verbose);
        int[] distances0 = result0.getDistanceMatrix();
        int v = set.getNumVertex();

        for(int a = 1; a < env.length;  a++){
            String execEnv1 = env[a][0];
            String algorithm1 = env[a][1];
            ApspResult<int[]> result1 = MatrixSetManager.getInstance().getApspResultInt(adjacencyFilename, execEnv1, algorithm1, blocks[a], verbose);
            int[] distances1 = result1.getDistanceMatrix();
            try{
                assertArrayEquals(distances0, distances1);
            }catch(Error err){
                for (int count = 1, i = 0; i < v; i++) {
                    for (int j = 0; j < v; j++) {
                        int index = i * v + j;
                        if (distances0[index] != distances1[index]) {
                            System.err.println(count +": "+ i + "," + j + " actual:" + distances0[index] + " expected:" + distances1[index]);
                            assertEquals(distances0[index], distances1[index]);
                            count++;
                        }
                        if(count >= 1000) {
                            // break;
                        }
                    }
                    if(count >= 1000) {
                        System.err.println("  ...cancel assertion");
                        break;
                    }
                }
            }
        }
    }
}
