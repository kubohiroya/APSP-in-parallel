package jp.ac.cuc.hiroya.apsp.util;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.ApspResult;

import java.io.IOException;

public class ApspMain {

    public static void resolveInt(String execEnv, String algorithm, String filename) throws IOException {
        int[] input_i = (filename == null)
                ? new int[] { 0, 10, 20, 30, 40, 10, 0, 30, 100, 100, 20, 30, 0, 40, 50, 30, 100, 40, 0, 60, 40, 100,
                        50, 60, 0, }
                : CSVParser.parseCSV_int(filename);
        ApspResult<int[]> result = ApspResolvers.IntResolver.resolve(execEnv, algorithm, input_i, 64);

        System.out.println("Process " + result.getNumVertex() + " x " + result.getNumVertex() + " nodes");
        System.out.println("Finished in " + result.getElapsedTime() + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_int(input_i, result.getNumVertex());
        System.out.println("[output]");
        ApspOutput.print_matrix_int(result.getOutput(), result.getNumVertex());
        System.out.println("[predecessors]");
        ApspOutput.print_matrix_int(result.getPredecessors(), result.getNumVertex());
    }

    public static void resolveFloat(String execEnv, String algorithm, String filename) throws IOException {
        float[] input_f = (filename == null)
                ? new float[] { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 10.0f, 0.0f, 30.0f, 100.0f, 100.0f, 20.0f, 30.0f,
                        0.0f, 40.0f, 50.0f, 30.0f, 100.0f, 40.0f, 0.0f, 60.0f, 40.0f, 100.0f, 50.0f, 60.0f, 0.0f, }
                : CSVParser.parseCSV_float(filename);

        ApspResult<float[]> result = ApspResolvers.FloatResolver.resolve(execEnv, algorithm, input_f, 64);

        System.out.println("Process " + result.getNumVertex() + " x " + result.getNumVertex() + " nodes");
        System.out.println("Finished in " + result.getElapsedTime() + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_float(input_f, result.getNumVertex());
        System.out.println("[output]");
        ApspOutput.print_matrix_float(result.getOutput(), result.getNumVertex());
        System.out.println("[predecessors]");
        ApspOutput.print_matrix_int(result.getPredecessors(), result.getNumVertex());
    }

    public static void resolveDouble(String execEnv, String algorithm, String filename) throws IOException {
        double[] input_d = (filename == null)
                ? new double[] { 0.0, 10.0, 20.0, 30.0, 40.0, 10.0, 0.0, 30.0, 100.0, 100.0, 20.0, 30.0, 0.0, 40.0,
                        50.0, 30.0, 100.0, 40.0, 0.0, 60.0, 40.0, 100.0, 50.0, 60.0, 0.0 }
                : CSVParser.parseCSV_double(filename);

        ApspResult<double[]> result = ApspResolvers.DoubleResolver.resolve(execEnv, algorithm, input_d, 64);

        System.out.println("Process " + result.getNumVertex() + " x " + result.getNumVertex() + " nodes");
        System.out.println("Finished in " + result.getElapsedTime() + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_double(input_d, result.getNumVertex());
        System.out.println("[output]");
        ApspOutput.print_matrix_double(result.getOutput(), result.getNumVertex());
        System.out.println("[predecessors]");
        ApspOutput.print_matrix_int(result.getPredecessors(), result.getNumVertex());
    }

    public static void main(String[] args) throws IOException {
        if (args.length <= 1) {
            System.err.println(
                    "java jp.ac.cuc.hiroya.apsp.util.ApspMain [seq|seq-ispc|omp|omp-ispc|cuda] [Floyd-Warshall|Johnson|f|j] [int|float|double|i|f|d] input.csv");
            System.exit(0);
        }
        String execEnv = args[0];
        String algorithm = args[1];
        String type = args[2];
        String filename = args.length != 4 ? null : args[3];

        switch (type) {
            case "int":
            case "i":
                resolveInt(execEnv, algorithm, filename);
                break;
            case "float":
            case "f":
                resolveFloat(execEnv, algorithm, filename);
                break;
            case "double":
            case "d":
            default:
                resolveDouble(execEnv, algorithm, filename);
                break;
        }
        System.exit(0);
    }
}
