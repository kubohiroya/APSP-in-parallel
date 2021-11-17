package jp.ac.cuc.hiroya.apsp.util;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.ApspResult;
import jp.ac.cuc.hiroya.apsp.lib.Infinity;

import java.io.IOException;

public class ApspMain {

    public static void resolve(String execEnv, String algorithm, String distanceMatrixMode, int[] adjacencyMatrix) throws IOException {
        int numVertex = (int) Math.sqrt(adjacencyMatrix.length);
        ApspResult<int[]> result = ApspResolvers.IntResolver.resolve(execEnv, algorithm, adjacencyMatrix, numVertex, 16);

        if(distanceMatrixMode.contains("time")) {
            System.out.println("Process " + result.getNumVertex() + " x " + result.getNumVertex() + " nodes");
            System.out.println("Finished in " + result.getElapsedTime() + " ms");
        }
        if(distanceMatrixMode.contains("distance")) {
            CSVOutput.print(result.getDistanceMatrix(), result.getNumVertex());
        }
        if(distanceMatrixMode.contains("node")) {
            CSVOutput.print(result.getSuccessorMatrix(), result.getNumVertex());
        }
    }

    public static void resolve(String execEnv, String algorithm, String distanceMatrixMode, float[] adjacencyMatrix) throws IOException {
        int numVertex = (int) Math.sqrt(adjacencyMatrix.length);
        ApspResult<float[]> result = ApspResolvers.FloatResolver.resolve(execEnv, algorithm, adjacencyMatrix, numVertex, 16);

        if(distanceMatrixMode.contains("time")) {
            System.out.println("Process " + result.getNumVertex() + " x " + result.getNumVertex() + " nodes");
            System.out.println("Finished in " + result.getElapsedTime() + " ms");
        }
        if(distanceMatrixMode.contains("distance")) {
            CSVOutput.print(result.getDistanceMatrix(), result.getNumVertex());
        }
        if(distanceMatrixMode.contains("node")) {
            CSVOutput.print(result.getSuccessorMatrix(), result.getNumVertex());
        }
    }

    public static void resolve(String execEnv, String algorithm, String distanceMatrixMode, double[] adjacencyMatrix) throws IOException {
        int numVertex = (int) Math.sqrt(adjacencyMatrix.length);
        ApspResult<double[]> result = ApspResolvers.DoubleResolver.resolve(execEnv, algorithm, adjacencyMatrix, numVertex, 16);

        if(distanceMatrixMode.contains("time")) {
            System.out.println("Process " + result.getNumVertex() + " x " + result.getNumVertex() + " nodes");
            System.out.println("Finished in " + result.getElapsedTime() + " ms");
        }
        if(distanceMatrixMode.contains("distance")) {
            CSVOutput.print(result.getDistanceMatrix(), result.getNumVertex());
        }
        if(distanceMatrixMode.contains("node")) {
            CSVOutput.print(result.getSuccessorMatrix(), result.getNumVertex());
        }
    }

    interface DemoData{
        int[] adjMatrixInt = new int[] { 0, 10, 20, 30, 40, 10, 0, 30, 100, 100, 20, 30, 0, 40, 50, 30, 100, 40, 0, 60, 40, 100,
                50, 60, 0, };

        float[] adjMatrixFloat = new float[] { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 10.0f, 0.0f, 30.0f, 100.0f, 100.0f, 20.0f, 30.0f,
                0.0f, 40.0f, 50.0f, 30.0f, 100.0f, 40.0f, 0.0f, 60.0f, 40.0f, 100.0f, 50.0f, 60.0f, 0.0f, };

        double[] adjMatrixDouble = new double[] { 0.0, 10.0, 20.0, 30.0, 40.0, 10.0, 0.0, 30.0, 100.0, 100.0, 20.0, 30.0, 0.0, 40.0,
                50.0, 30.0, 100.0, 40.0, 0.0, 60.0, 40.0, 100.0, 50.0, 60.0, 0.0 };
    }

    public static void main(String[] args) throws IOException {
        if (args.length <= 1) {
            System.err.println(
                    "java jp.ac.cuc.hiroya.apsp.util.ApspMain"+
                            " [seq|seq-ispc|omp|omp-ispc|cuda] [Floyd-Warshall|Johnson|f|j] [int|float|double|i|f|d] [time|distance|node] adjacencyMatrix.csv");
            System.exit(0);
        }
        String execEnv = args[0];
        String algorithm = args[1];
        String distanceType = args[2];
        String distanceMatrixMode = args.length < 4 ? "time" : args[3];
        String filename = args.length != 5 ? null : args[4];

        switch (distanceType) {
            case "int":
            case "i":
                int[] adjMatrixInt = (filename == null)
                        ? DemoData.adjMatrixInt
                        : InfinityConverter.convert(CSVParser.parseIntCSV(filename), 0, Infinity.INT_INF);
                resolve(execEnv, algorithm, distanceMatrixMode, adjMatrixInt);
                break;
            case "float":
            case "f":
                float[] adjMatrixFloat = (filename == null)
                        ? DemoData.adjMatrixFloat
                        : InfinityConverter.convert(CSVParser.parseFloatCSV(filename), 0.0f, Infinity.FLT_INF);
                resolve(execEnv, algorithm, distanceMatrixMode, adjMatrixFloat);
                break;
            case "double":
            case "d":
            default:
                double[] adjMatrixDouble = (filename == null)
                        ? DemoData.adjMatrixDouble
                        : InfinityConverter.convert(CSVParser.parseDoubleCSV(filename), 0.0, Infinity.DBL_INF);
                resolve(execEnv, algorithm, distanceMatrixMode, adjMatrixDouble);
                break;
        }
        System.exit(0);
    }
}
