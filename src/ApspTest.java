import com.sun.jna.Library;
import com.sun.jna.Native;

public class ApspTest {

    public interface ApspLibrary extends Library {
        public void nop();

        public void floyd_warshall_blocked(int[] input, int[] output, int[] parents, int n, int b);

        public void floyd_warshall_blocked_float(float[] input, float[] output, int[] parents, int n, int b);

        public void floyd_warshall_blocked_double(double[] input, double[] output, int[] parents, int n, int b);

        public void johnson_parallel_matrix(int[] input, int[] output, int[] parents, int n);

        public void johnson_parallel_matrix_float(float[] input, float[] output, int[] parents, int n);

        public void johnson_parallel_matrix_double(double[] input, double[] output, int[] parents, int n);
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

    static void exec_int(int[] input, int n, String algorithm) {
        int[] output = new int[n * n];
        int[] parents = new int[n * n];

        long timeStart = System.currentTimeMillis();

        switch (algorithm) {
            case "seq-f":
                ApspSeq.INSTANCE.floyd_warshall_blocked(input, output, parents, n, 1);
                break;
            case "seq-ispc-f":
                ApspSeqIspc.INSTANCE.floyd_warshall_blocked(input, output, parents, n, 1);
                break;
            case "omp-f":
                ApspOmp.INSTANCE.floyd_warshall_blocked(input, output, parents, n, 1);
                break;
            case "omp-ispc-f":
                ApspOmpIspc.INSTANCE.floyd_warshall_blocked(input, output, parents, n, 1);
                break;
            case "seq-j":
                ApspSeq.INSTANCE.johnson_parallel_matrix(input, output, parents, n);
                break;
            case "seq-ispc-j":
                ApspSeqIspc.INSTANCE.johnson_parallel_matrix(input, output, parents, n);
                break;
            case "omp-j":
                ApspOmp.INSTANCE.johnson_parallel_matrix(input, output, parents, n);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.johnson_parallel_matrix(input, output, parents, n);
                break;
            default:
                ApspSeq.INSTANCE.nop();
                break;
        }
        long timeEnd = System.currentTimeMillis();

        System.out.println("Finished in " + (timeEnd - timeStart) + " ms");

        System.out.println("[input]");
        print_matrix_int(input, n);
        System.out.println("[output]");
        print_matrix_int(output, n);
        System.out.println("[parents]");
        print_matrix_int(parents, n);
    }

    static void exec_float(float[] input, int n, String algorithm) {
        float[] output = new float[n * n];
        int[] parents = new int[n * n];

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
            case "omp-j":
                ApspOmp.INSTANCE.johnson_parallel_matrix_float(input, output, parents, n);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.johnson_parallel_matrix_float(input, output, parents, n);
                break;
        }
        long timeEnd = System.currentTimeMillis();

        System.out.println("Finished in " + (timeEnd - timeStart) + " ms");

        System.out.println("[input]");
        print_matrix_float(input, n);
        System.out.println("[output]");
        print_matrix_float(output, n);
        System.out.println("[parents]");
        print_matrix_int(parents, n);
    }

    static void exec_double(double[] input, int n, String algorithm) {
        double[] output = new double[n * n];
        int[] parents = new int[n * n];

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
            case "omp-j":
                ApspOmp.INSTANCE.johnson_parallel_matrix_double(input, output, parents, n);
                break;
            case "omp-ispc-j":
                ApspOmpIspc.INSTANCE.johnson_parallel_matrix_double(input, output, parents, n);
                break;
        }
        long timeEnd = System.currentTimeMillis();

        System.out.println("Finished in " + (timeEnd - timeStart) + " ms");

        System.out.println("[input]");
        print_matrix_double(input, n);
        System.out.println("[output]");
        print_matrix_double(output, n);
        System.out.println("[parents]");
        print_matrix_int(parents, n);
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("java ApspTest [seq-f|seq-ispc-f|omp-f|omp-ispc-f|seq-j|seq-ispc-j|omp-j|omp-ispc-j] [i|f|d]");
            System.exit(0);
        }
        String algorithm = args[0];
        String type = args[1];
        int n = 5;
        int[] input_i = {
                0, 10, 200, 30, 40,
                10, 0, 30, 100, 100,
                20, 30, 0, 40, 50,
                30, 100, 40, 0, 60,
                40, 100, 50, 60, 0,
        };
        float[] input_f = {
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        };
        double[] input_d = {
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0
        };
        switch (type) {
            case "i":
                exec_int(input_i, n, algorithm);
                break;
            case "f":
                exec_float(input_f, n, algorithm);
                break;
            case "d":
                exec_double(input_d, n, algorithm);
                break;
        }
    }
}
