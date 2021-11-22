package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import jp.ac.cuc.hiroya.apsp.util.CSVOutput;

import java.util.Locale;
import java.util.Random;

public class RandomMatrixGenerator {

    public static double[] generateRandomAdjacencyMatrix(final long seed, final int n, final double p, final double min, final double max, final double inf){
        double adjacencyMatrix[] = new double[n * n];
        Random rand = new Random(seed);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                int index = i * n + j;
                if(i == j){
                    adjacencyMatrix[index] = 0.0;
                }else if(j > i){
                    if(rand.nextDouble() < p) {
                        adjacencyMatrix[index] = min + rand.nextDouble() * (max - min);
                    }else {
                        adjacencyMatrix[index] = inf;
                    }
                }else{
                    adjacencyMatrix[index] = adjacencyMatrix[j * n + i];
                }
            }
        }
        // CSVOutput.print(adjacencyMatrix, n);
        return adjacencyMatrix;
    }

    public static float[] generateRandomAdjacencyMatrix(final long seed, final int n, final double p, final float min, final float max, final float inf){
        float adjacencyMatrix[] = new float[n * n];
        Random rand = new Random(seed);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                int index = i * n + j;
                if(i == j){
                    adjacencyMatrix[index] = 0f;
                }else if(i < j){
                    if(rand.nextDouble() < p) {
                        adjacencyMatrix[index] = min + (float) (rand.nextDouble() * (max - min));
                    }else {
                        adjacencyMatrix[index] = inf;
                    }
                }else{
                    adjacencyMatrix[index] = adjacencyMatrix[j * n + i];
                }
            }
        }
        // CSVOutput.print(adjacencyMatrix, n);
        return adjacencyMatrix;
    }

    public static int[] generateRandomAdjacencyMatrix(final long seed, final int n, final double p, final int min, final int max, final int inf){
        int adjacencyMatrix[] = new int[n * n];
        Random rand = new Random(seed);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                int index = i * n + j;
                if(i == j){
                    adjacencyMatrix[index] = 0;
                }else if(i < j){
                    if(rand.nextDouble() < p) {
                        adjacencyMatrix[index] = min + (int) (rand.nextDouble() * (max - min));
                    }else {
                        adjacencyMatrix[index] = inf;
                    }
                }else{
                    adjacencyMatrix[index] = adjacencyMatrix[j * n + i];
                }
            }
        }
        // CSVOutput.print(adjacencyMatrix, n);
        return adjacencyMatrix;
    }

    public static void main(String[] args)throws Exception{
        if(args.length < 7){
            System.out.println("Usage:");
            System.out.println("java jp.ac.cuc.hiroya.apsp.RandomMatrixGenerator SEED N P MIN MAX [0|\"Inf\"] [double|float|int|d|f|i]");
            System.exit(0);
        }
        long seed = Long.parseLong(args[0]);
        int n = Integer.parseInt(args[1]);
        double p = Double.parseDouble(args[2]);
        double min = Double.parseDouble(args[3]);
        double max = Double.parseDouble(args[4]);
        boolean inf = args[5].toLowerCase(Locale.ROOT).equals("inf");
        String type = args[6];
        switch(type){
            case "double":
            case "d":
                CSVOutput.print(RandomMatrixGenerator.generateRandomAdjacencyMatrix(seed, n, p, min, max, inf?Infinity.DBL_INF:0.0), n);
                break;
            case "float":
            case "f":
                CSVOutput.print(RandomMatrixGenerator.generateRandomAdjacencyMatrix(seed, n, p, (float)min, (float)max, inf?Infinity.FLT_INF:0f), n);
                break;
            case "int":
            case "i":
                CSVOutput.print(RandomMatrixGenerator.generateRandomAdjacencyMatrix(seed, n, p, (int)min, (int)max, inf?Infinity.INT_INF:0), n);
                break;
        }
        System.exit(0);
    }
}
