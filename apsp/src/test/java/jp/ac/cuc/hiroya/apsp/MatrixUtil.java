package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import jp.ac.cuc.hiroya.apsp.util.CSVParser;
import jp.ac.cuc.hiroya.apsp.util.InfinityConverter;
import jp.ac.cuc.hiroya.apsp.util.SuccessorNormalizer;

import java.io.IOException;

public class MatrixUtil {

    static String PREFIX = "random:";

    public static double[] getAdjacencyMatrix(String adjacencyFilename, double inf)throws IOException, NumberFormatException{
        if(adjacencyFilename == null){
            return null;
        }
        if(adjacencyFilename.startsWith(PREFIX)) {
            String[] args = adjacencyFilename.substring(PREFIX.length()).split(",");
            long seed = Long.parseLong(args[0]);
            int n = Integer.parseInt(args[1]);
            double p = Double.parseDouble(args[2]);
            double min = Double.parseDouble(args[3]);
            double max = Double.parseDouble(args[4]);
            return RandomMatrixGenerator.generateRandomAdjacencyMatrix(seed, n, p, min, max, inf);
        }
        return MatrixUtil.loadCsvFileDouble(adjacencyFilename);
    }

    public static float[] getAdjacencyMatrix(String adjacencyFilename, float inf)throws IOException, NumberFormatException{
        if(adjacencyFilename == null){
            return null;
        }
        if(adjacencyFilename.startsWith(PREFIX)) {
            String[] args = adjacencyFilename.substring(PREFIX.length()).split(",");
            long seed = Long.parseLong(args[0]);
            int n = Integer.parseInt(args[1]);
            double p = Double.parseDouble(args[2]);
            float min = Float.parseFloat(args[3]);
            float max = Float.parseFloat(args[4]);
            return RandomMatrixGenerator.generateRandomAdjacencyMatrix(seed, n, p, min, max, inf);
        }
        return MatrixUtil.loadCsvFileFloat(adjacencyFilename);
    }

    public static int[] getAdjacencyMatrix(String adjacencyFilename, int inf)throws IOException, NumberFormatException{
        if(adjacencyFilename == null){
            return null;
        }
        if(adjacencyFilename.startsWith(PREFIX)) {
            String[] args = adjacencyFilename.substring(PREFIX.length()).split(",");
            long seed = Long.parseLong(args[0]);
            int n = Integer.parseInt(args[1]);
            double p = Double.parseDouble(args[2]);
            int min = Integer.parseInt(args[3]);
            int max = Integer.parseInt(args[4]);
            return RandomMatrixGenerator.generateRandomAdjacencyMatrix(seed, n, p, min, max, inf);
        }
        return MatrixUtil.loadCsvFileInt(adjacencyFilename);
    }

    static String value(double v){
        return v == Infinity.DBL_INF? "Inf" : Double.toString(v);
    }
    static String value(float v){
        return v == Infinity.FLT_INF? "Inf" : Float.toString(v);
    }
    static String value(int v){
        return v == Infinity.INT_INF? "Inf" : Integer.toString(v);
    }

    public static double calculateDistance(int from, int to, int v, double[] adjacencyMatrix, int[] successorMatrix, boolean verbose){
        if(verbose) System.out.println(from+"発 => "+to+"行");
        double distance = 0;
        if(from == to){
            return distance;
        }
        while(true){
            int next = successorMatrix[from * v + to];
            distance += adjacencyMatrix[from * v + next];
            if(verbose) System.out.println("   "+from+"発 => "+next+"行 \t+ "+value(adjacencyMatrix[from * v + next]));
            if(from == next){
                distance = Infinity.DBL_INF;
                break;
            }
            if(next == to){
                if(verbose) System.out.println("    合計距離 = "+value(distance));
                break;
            }
            from = next;
        }
        return distance;
    }

    public static float calculateDistance(int from, int to, int v, float[] adjacencyMatrix, int[] successorMatrix, boolean verbose){
        if(verbose) System.out.println(from+"発 => "+to+"行");
        float distance = 0;
        if(from == to){
            return distance;
        }
        while(true){
            int next = successorMatrix[from * v + to];
            distance += adjacencyMatrix[from * v + next];
            if(verbose) System.out.println("   "+from+"発 => "+next+"行 \t+ "+value(adjacencyMatrix[from * v + next]));
            if(from == next){
                distance = Infinity.FLT_INF;
                break;
            }
            if(next == to){
                if(verbose) System.out.println("    合計距離 = "+value(distance));
                break;
            }
            from = next;
        }
        return distance;
    }

    public static int calculateDistance(int from, int to, int v, int[] adjacencyMatrix, int[] successorMatrix, boolean verbose){
        if(verbose) System.out.println(from+"発 => "+to+"行");
        int distance = 0;
        if(from == to){
            return distance;
        }
        while(true){
            int next = successorMatrix[from * v + to];
            distance += adjacencyMatrix[from * v + next];
            if(verbose) System.out.println("   "+from+"発 => "+next+"行 \t+ "+value(adjacencyMatrix[from * v + next]));
            if(from == next){
                distance = Infinity.INT_INF;
                break;
            }
            if(next == to){
                if(verbose) System.out.println("    合計距離 = "+value(distance));
                break;
            }
            from = next;
        }
        return distance;
    }


    static double[] calculateDistanceMatrix(double[] adjacencyMatrix, int[] successorMatrix, boolean verbose){
        int v = (int) Math.sqrt(adjacencyMatrix.length);
        double[] distanceMatrix = new double[v * v];
        for(int i = 0; i < v; i++){
            for(int j = 0; j < v; j++){
                distanceMatrix[i*v + j] = calculateDistance(i, j, v, adjacencyMatrix, successorMatrix, verbose);
            }
        }
        return distanceMatrix;
    }

    static float[] calculateDistanceMatrix(float[] adjacencyMatrix, int[] successorMatrix, boolean verbose){
        int v = (int) Math.sqrt(adjacencyMatrix.length);
        float[] distanceMatrix = new float[v * v];
        for(int i = 0; i < v; i++){
            for(int j = 0; j < v; j++){
                distanceMatrix[i*v + j] = calculateDistance(i, j, v, adjacencyMatrix, successorMatrix, verbose);
            }
        }
        return distanceMatrix;
    }

    static int[] calculateDistanceMatrix(int[] adjacencyMatrix, int[] successorMatrix, boolean verbose){
        int v = (int) Math.sqrt(adjacencyMatrix.length);
        int[] distanceMatrix = new int[v * v];
        for(int i = 0; i < v; i++){
            for(int j = 0; j < v; j++){
                distanceMatrix[i*v + j] = calculateDistance(i, j, v, adjacencyMatrix, successorMatrix, verbose);
            }
        }
        return distanceMatrix;
    }

    static double[] loadCsvFileDouble(String filename)throws IOException {
        return InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
    }

    static float[] loadCsvFileFloat(String filename)throws IOException {
        return InfinityConverter.convert(CSVParser.parseFloatCSV(filename), 0.0f, Infinity.FLT_INF);
    }

    static int[] loadCsvFileInt(String filename)throws IOException {
        return InfinityConverter.convert(CSVParser.parseIntCSV(filename), 0, Infinity.INT_INF);
    }

    static int[] loadSuccessorMatrix(String nodeFilename)throws IOException{
        return SuccessorNormalizer.normalize(CSVParser.parseIntCSV(nodeFilename));
    }

}
