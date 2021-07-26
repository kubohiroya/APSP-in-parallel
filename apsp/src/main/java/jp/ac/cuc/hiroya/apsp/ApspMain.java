package jp.ac.cuc.hiroya.apsp;

import java.io.IOException;

public class ApspMain {

    public static void resolveInt(String filename, String execEnv, String algorithm) throws IOException {
        int[] input_i = (filename == null)
                ? new int[] { 0, 10, 20, 30, 40, 10, 0, 30, 100, 100, 20, 30, 0, 40, 50, 30, 100, 40, 0, 60, 40, 100,
                        50, 60, 0, }
                : CSVParser.parseCSV_int(filename);
        ApspResult<int[]> result = ApspResolvers.IntResolver.resolve(execEnv, algorithm, input_i, 64);

        System.out.println("Process " + result.numVertex + " x " + result.numVertex + " nodes");
        System.out.println("Finished in " + result.elapsedTime + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_int(input_i, result.numVertex);
        System.out.println("[output]");
        ApspOutput.print_matrix_int(result.output, result.numVertex);
        System.out.println("[parents]");
        ApspOutput.print_matrix_int(result.parents, result.numVertex);
    }

    public static void resolveFloat(String filename, String execEnv, String algorithm) throws IOException {
        float[] input_f = (filename == null)
                ? new float[] { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 10.0f, 0.0f, 30.0f, 100.0f, 100.0f, 20.0f, 30.0f,
                        0.0f, 40.0f, 50.0f, 30.0f, 100.0f, 40.0f, 0.0f, 60.0f, 40.0f, 100.0f, 50.0f, 60.0f, 0.0f, }
                : CSVParser.parseCSV_float(filename);

        ApspResult<float[]> result = ApspResolvers.FloatResolver.resolve(execEnv, algorithm, input_f, 64);

        System.out.println("Process " + result.numVertex + " x " + result.numVertex + " nodes");
        System.out.println("Finished in " + result.elapsedTime + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_float(input_f, result.numVertex);
        System.out.println("[output]");
        ApspOutput.print_matrix_float(result.output, result.numVertex);
        System.out.println("[parents]");
        ApspOutput.print_matrix_int(result.parents, result.numVertex);
    }

    public static void resolveDouble(String filename, String execEnv, String algorithm) throws IOException {
        double[] input_d = (filename == null)
                ? new double[] { 0.0, 10.0, 20.0, 30.0, 40.0, 10.0, 0.0, 30.0, 100.0, 100.0, 20.0, 30.0, 0.0, 40.0,
                        50.0, 30.0, 100.0, 40.0, 0.0, 60.0, 40.0, 100.0, 50.0, 60.0, 0.0 }
                : CSVParser.parseCSV_double(filename);

        ApspResult<double[]> result = ApspResolvers.DoubleResolver.resolve(execEnv, algorithm, input_d, 64);

        System.out.println("Process " + result.numVertex + " x " + result.numVertex + " nodes");
        System.out.println("Finished in " + result.elapsedTime + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_double(input_d, result.numVertex);
        System.out.println("[output]");
        ApspOutput.print_matrix_double(result.output, result.numVertex);
        System.out.println("[parents]");
        ApspOutput.print_matrix_int(result.parents, result.numVertex);
    }

    public static void main(String[] args) throws IOException {
        if (args.length <= 1) {
            System.err.println(
                    "java ApspMain [seq|seq-ispc|omp|omp-ispc|cuda] [Floyd-Warshall|johnson] [int|float|double] input.csv");
            System.exit(0);
        }
        String execEnv = args[0];
        String algorithm = args[1];
        String type = args[2];
        String filename = args.length != 4 ? null : args[3];

        switch (type) {
            case "int":
            case "i":
                resolveInt(filename, execEnv, algorithm);
                break;
            case "float":
            case "f":
                resolveFloat(filename, execEnv, algorithm);
                break;
            case "double":
            case "d":
            default:
                resolveDouble(filename, execEnv, algorithm);
                break;
        }
        System.exit(0);
    }
}
