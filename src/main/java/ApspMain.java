import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import com.sun.jna.Memory;
import java.io.*;

class ApspOutput {
    static final int INT_INF = 1073741823;
    static final float FLT_INF = 3.402823466e+37f;
    static final double DBL_INF = 1.7976931348623158e+307;

    public static void print_int(int value) {
        System.out.print(value == INT_INF ? "Inf" : value);
    }

    public static void print_float(float value) {
        System.out.print(value == FLT_INF ? "Inf" : value);
    }

    public static void print_double(double value) {
        System.out.print(value == DBL_INF ? "Inf" : value);
    }

    public static void print_matrix_int(int[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            print_int(matrix[i * n]);
            for (int j = 1; j < n; j++) {
                System.out.print(", ");
                print_int(matrix[i * n + j]);
            }
            System.out.println();
        }
    }

    public static void print_matrix_float(float[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            print_float(matrix[i * n]);
            for (int j = 1; j < n; j++) {
                System.out.print(", ");
                print_float(matrix[i * n + j]);
            }
            System.out.println();
        }
    }

    public static void print_matrix_double(double[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            print_double(matrix[i * n]);
            for (int j = 1; j < n; j++) {
                System.out.print(", ");
                print_double(matrix[i * n + j]);
            }
            System.out.println();
        }
    }
}

class CSVParser {

    public static int[] parseCSV_int(String filename) throws IOException {
            int n = 0;
            int[] ret = null;
            int p = 0;
            File file = new File(filename);
            BufferedReader br = new BufferedReader(new FileReader(file));
            String line;
            try {
                while ((line = br.readLine()) != null) {
                    String[] row = line.split(",");
                    if (ret == null) {
                        n = row.length;
                        ret = new int[n * n];
                    }
                    for (int i = 0; i < row.length; i++) {
                        ret[p++] = (int) Float.parseFloat(row[i]);
                    }
                }
            } finally {
                br.close();
            }
            return ret;
        }

    public static float[] parseCSV_float(String filename) throws IOException {
        int n = 0;
        float[] ret = null;
        int p = 0;
        File file = new File(filename);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        try {
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    n = row.length;
                    ret = new float[n * n];
                }
                for (int i = 0; i < row.length; i++) {
                    ret[p++] = Float.parseFloat(row[i]);
                }
            }
        } finally {
            br.close();
        }
        return ret;
    }

    public static double[] parseCSV_double(String filename) throws IOException {
        int n = 0;
        double[] ret = null;
        int p = 0;
        File file = new File(filename);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        try {
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    n = row.length;
                    ret = new double[n * n];
                }
                for (int i = 0; i < row.length; i++) {
                    ret[p++] = Double.parseDouble(row[i]);
                }
            }
        } finally {
            br.close();
        }
        return ret;
    }
}

class ApspResult<T> {
    T output;
    int[] parents;
    long elapsedTime;

    ApspResult(T output, int[] parents, long elapsedTime) {
        this.output = output;
        this.parents = parents;
        this.elapsedTime = elapsedTime;
    }
}

public class ApspMain {

    public interface ApspLibrary extends Library {
        public void floyd_warshall_blocked_int(int[] input, PointerByReference output, PointerByReference parents,
                int n, int b);

        public void free_floyd_warshall_blocked_int(PointerByReference output, PointerByReference parents);

        public void floyd_warshall_blocked_float(float[] input, PointerByReference output, PointerByReference parents,
                int n, int b);

        public void free_floyd_warshall_blocked_float(PointerByReference output, PointerByReference parents);

        public void floyd_warshall_blocked_double(double[] input, PointerByReference output, PointerByReference parents,
                int n, int b);

        public void free_floyd_warshall_blocked_double(PointerByReference output, PointerByReference parents);

        public void johnson_parallel_matrix_int(int[] input, PointerByReference output, PointerByReference parents,
                int n);

        public void free_johnson_parallel_matrix_int(PointerByReference output, PointerByReference parents);

        public void johnson_parallel_matrix_float(float[] input, PointerByReference output, PointerByReference parents,
                int n);

        public void free_johnson_parallel_matrix_float(PointerByReference output, PointerByReference parents);

        public void johnson_parallel_matrix_double(double[] input, PointerByReference output,
                PointerByReference parents, int n);

        public void free_johnson_parallel_matrix_double(PointerByReference output, PointerByReference parents);
    }

    public interface ApspSeq extends ApspLibrary {
        ApspSeq INSTANCE = Native.load("apsp-seq", ApspSeq.class);
    }

    public interface ApspSeqIspc extends ApspLibrary {
        ApspSeqIspc INSTANCE = Native.load("apsp-seq-ispc", ApspSeqIspc.class);
    }

    public interface ApspOmp extends ApspLibrary {
        ApspOmp INSTANCE = Native.load("apsp-omp", ApspOmp.class);
    }

    public interface ApspOmpIspc extends ApspLibrary {
        ApspOmpIspc INSTANCE = Native.load("apsp-omp-ispc", ApspOmpIspc.class);
    }

    static ApspResult<int[]> exec_apsp_algorithm_int(int[] input, int n, String algorithm) {
        PointerByReference output = new PointerByReference();
        PointerByReference parents = new PointerByReference();
        long timeStart = System.currentTimeMillis();
        switch (algorithm) {
            case "seq-f":
                ApspSeq.INSTANCE.floyd_warshall_blocked_int(input, output, parents, n, 1);
                break;
            case "seq-ispc-f":
                ApspSeqIspc.INSTANCE.floyd_warshall_blocked_int(input, output, parents, n, 1);
                break;
            case "omp-f":
                ApspOmp.INSTANCE.floyd_warshall_blocked_int(input, output, parents, n, 1);
                break;
            case "omp-ispc-f":
                ApspOmpIspc.INSTANCE.floyd_warshall_blocked_int(input, output, parents, n, 1);
                break;
            case "seq-j":
                ApspSeq.INSTANCE.johnson_parallel_matrix_int(input, output, parents, n);
                break;
            case "seq-ispc-j":
                ApspSeqIspc.INSTANCE.johnson_parallel_matrix_int(input, output, parents, n);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.johnson_parallel_matrix_int(input, output, parents, n);
                break;
            case "omp-j":
            default:
                ApspOmp.INSTANCE.johnson_parallel_matrix_int(input, output, parents, n);
                break;
        }
        long timeEnd = System.currentTimeMillis();
        int[] outputMatrix = output.getValue().getIntArray(0, n * n);
        int[] parentMatrix = parents.getValue().getIntArray(0, n * n);
        switch (algorithm) {
            case "seq-f":
                ApspSeq.INSTANCE.free_floyd_warshall_blocked_int(output, parents);
                break;
            case "seq-ispc-f":
                ApspSeqIspc.INSTANCE.free_floyd_warshall_blocked_int(output, parents);
                break;
            case "omp-f":
                ApspOmp.INSTANCE.free_floyd_warshall_blocked_int(output, parents);
                break;
            case "omp-ispc-f":
                ApspOmpIspc.INSTANCE.free_floyd_warshall_blocked_int(output, parents);
                break;
            case "seq-j":
                ApspSeq.INSTANCE.free_johnson_parallel_matrix_int(output, parents);
                break;
            case "seq-ispc-j":
                ApspSeqIspc.INSTANCE.free_johnson_parallel_matrix_int(output, parents);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.free_johnson_parallel_matrix_int(output, parents);
                break;
            case "omp-j":
            default:
                ApspOmp.INSTANCE.free_johnson_parallel_matrix_int(output, parents);
                break;
        }
        return new ApspResult<int[]>(outputMatrix, parentMatrix, timeEnd - timeStart);
    }

    static ApspResult<float[]> exec_apsp_algorithm_float(float[] input, int n, String algorithm) {
        PointerByReference output = new PointerByReference();
        PointerByReference parents = new PointerByReference();
        long timeStart = System.currentTimeMillis();
        switch (algorithm) {
            case "seq-f":
                ApspSeq.INSTANCE.floyd_warshall_blocked_float(input, output, parents, n, 1);
                break;
            case "seq-ispc-f":
                ApspSeqIspc.INSTANCE.floyd_warshall_blocked_float(input, output, parents, n, 1);
                break;
            case "omp-f":
                ApspOmp.INSTANCE.floyd_warshall_blocked_float(input, output, parents, n, 1);
                break;
            case "omp-ispc-f":
                ApspOmpIspc.INSTANCE.floyd_warshall_blocked_float(input, output, parents, n, 1);
                break;
            case "seq-j":
                ApspSeq.INSTANCE.johnson_parallel_matrix_float(input, output, parents, n);
                break;
            case "seq-ispc-j":
                ApspSeqIspc.INSTANCE.johnson_parallel_matrix_float(input, output, parents, n);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.johnson_parallel_matrix_float(input, output, parents, n);
                break;
            case "omp-j":
            default:
                ApspOmp.INSTANCE.johnson_parallel_matrix_float(input, output, parents, n);
                break;
        }
        long timeEnd = System.currentTimeMillis();
        float[] outputMatrix = output.getValue().getFloatArray(0, n * n);
        int[] parentMatrix = parents.getValue().getIntArray(0, n * n);
        switch (algorithm) {
            case "seq-f":
                ApspSeq.INSTANCE.free_floyd_warshall_blocked_float(output, parents);
                break;
            case "seq-ispc-f":
                ApspSeqIspc.INSTANCE.free_floyd_warshall_blocked_float(output, parents);
                break;
            case "omp-f":
                ApspOmp.INSTANCE.free_floyd_warshall_blocked_float(output, parents);
                break;
            case "omp-ispc-f":
                ApspOmpIspc.INSTANCE.free_floyd_warshall_blocked_float(output, parents);
                break;
            case "seq-j":
                ApspSeq.INSTANCE.free_johnson_parallel_matrix_float(output, parents);
                break;
            case "seq-ispc-j":
                ApspSeqIspc.INSTANCE.free_johnson_parallel_matrix_float(output, parents);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.free_johnson_parallel_matrix_float(output, parents);
                break;
            case "omp-j":
            default:
                ApspOmp.INSTANCE.free_johnson_parallel_matrix_float(output, parents);
                break;
        }
        return new ApspResult<float[]>(outputMatrix, parentMatrix, timeEnd - timeStart);
    }

    static ApspResult<double[]> exec_apsp_algorithm_double(double[] input, int n, String algorithm) {
        PointerByReference output = new PointerByReference();
        PointerByReference parents = new PointerByReference();
        long timeStart = System.currentTimeMillis();
        switch (algorithm) {
            case "seq-f":
                ApspSeq.INSTANCE.floyd_warshall_blocked_double(input, output, parents, n, 1);
                break;
            case "seq-ispc-f":
                ApspSeqIspc.INSTANCE.floyd_warshall_blocked_double(input, output, parents, n, 1);
                break;
            case "omp-f":
                ApspOmp.INSTANCE.floyd_warshall_blocked_double(input, output, parents, n, 1);
                break;
            case "omp-ispc-f":
                ApspOmpIspc.INSTANCE.floyd_warshall_blocked_double(input, output, parents, n, 1);
                break;
            case "seq-j":
                ApspSeq.INSTANCE.johnson_parallel_matrix_double(input, output, parents, n);
                break;
            case "seq-ispc-j":
                ApspSeqIspc.INSTANCE.johnson_parallel_matrix_double(input, output, parents, n);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.johnson_parallel_matrix_double(input, output, parents, n);
                break;
            case "omp-j":
            default:
                ApspOmp.INSTANCE.johnson_parallel_matrix_double(input, output, parents, n);
                break;
        }
        long timeEnd = System.currentTimeMillis();
        double[] outputMatrix = output.getValue().getDoubleArray(0, n * n);
        int[] parentMatrix = parents.getValue().getIntArray(0, n * n);
        switch (algorithm) {
            case "seq-f":
                ApspSeq.INSTANCE.free_floyd_warshall_blocked_double(output, parents);
                break;
            case "seq-ispc-f":
                ApspSeqIspc.INSTANCE.free_floyd_warshall_blocked_double(output, parents);
                break;
            case "omp-f":
                ApspOmp.INSTANCE.free_floyd_warshall_blocked_double(output, parents);
                break;
            case "omp-ispc-f":
                ApspOmpIspc.INSTANCE.free_floyd_warshall_blocked_double(output, parents);
                break;
            case "seq-j":
                ApspSeq.INSTANCE.free_johnson_parallel_matrix_double(output, parents);
                break;
            case "seq-ispc-j":
                ApspSeqIspc.INSTANCE.free_johnson_parallel_matrix_double(output, parents);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.free_johnson_parallel_matrix_double(output, parents);
                break;
            case "omp-j":
            default:
                ApspOmp.INSTANCE.free_johnson_parallel_matrix_double(output, parents);
                break;
        }
        return new ApspResult<double[]>(outputMatrix, parentMatrix, timeEnd - timeStart);
    }

    public static void run_apsp_int(String filename, String algorithm) throws IOException {
        int[] input_i = (filename == null)
                ? new int[] { 0, 10, 20, 30, 40, 10, 0, 30, 100, 100, 20, 30, 0, 40, 50, 30, 100, 40, 0, 60, 40, 100,
                        50, 60, 0, }
                : CSVParser.parseCSV_int(filename);
        int n = (int) Math.sqrt(input_i.length);

        ApspResult<int[]> result = exec_apsp_algorithm_int(input_i, n, algorithm);

        System.out.println("Process " + n + " x " + n + " nodes");
        System.out.println("Finished in " + result.elapsedTime + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_int(input_i, n);
        System.out.println("[output]");
        ApspOutput.print_matrix_int(result.output, n);
        System.out.println("[parents]");
        ApspOutput.print_matrix_int(result.parents, n);
    }

    public static void run_apsp_float(String filename, String algorithm) throws IOException {
        float[] input_f = (filename == null)
                ? new float[] { 0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 10.0f, 0.0f, 30.0f, 100.0f, 100.0f, 20.0f, 30.0f,
                        0.0f, 40.0f, 50.0f, 30.0f, 100.0f, 40.0f, 0.0f, 60.0f, 40.0f, 100.0f, 50.0f, 60.0f, 0.0f, }
                : CSVParser.parseCSV_float(filename);
        int n = (int) Math.sqrt(input_f.length);

        ApspResult<float[]> result = exec_apsp_algorithm_float(input_f, n, algorithm);

        System.out.println("Process " + n + " x " + n + " nodes");
        System.out.println("Finished in " + result.elapsedTime + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_float(input_f, n);
        System.out.println("[output]");
        ApspOutput.print_matrix_float(result.output, n);
        System.out.println("[parents]");
        ApspOutput.print_matrix_int(result.parents, n);
    }

    public static void run_apsp_double(String filename, String algorithm) throws IOException {
        double[] input_d = (filename == null)
                ? new double[] { 0.0, 10.0, 20.0, 30.0, 40.0, 10.0, 0.0, 30.0, 100.0, 100.0, 20.0, 30.0, 0.0, 40.0,
                        50.0, 30.0, 100.0, 40.0, 0.0, 60.0, 40.0, 100.0, 50.0, 60.0, 0.0 }
                : CSVParser.parseCSV_double(filename);
        int n = (int) Math.sqrt(input_d.length);

        ApspResult<double[]> result = exec_apsp_algorithm_double(input_d, n, algorithm);

        System.out.println("Process " + n + " x " + n + " nodes");
        System.out.println("Finished in " + result.elapsedTime + " ms");
        System.out.println("[input]");
        ApspOutput.print_matrix_double(input_d, n);
        System.out.println("[output]");
        ApspOutput.print_matrix_double(result.output, n);
        System.out.println("[parents]");
        ApspOutput.print_matrix_int(result.parents, n);
    }

    public static void main(String[] args) throws IOException {
        if (args.length <= 1) {
            System.err.println(
                    "java ApspMain [seq-f|seq-ispc-f|omp-f|omp-ispc-f|seq-j|seq-ispc-j|omp-j|omp-ispc-j] [i|f|d] input.csv");
            System.exit(0);
        }
        String algorithm = args[0];
        String type = args[1];
        String filename = args.length != 3 ? null : args[2];

        switch (type) {
            case "i":
                run_apsp_int(filename, algorithm);
                break;
            case "f":
                run_apsp_float(filename, algorithm);
                break;
            case "d":
            default:
                run_apsp_double(filename, algorithm);
                break;
        }
        System.exit(0);
    }
}
