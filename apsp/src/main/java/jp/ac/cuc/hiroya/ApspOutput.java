package jp.ac.cuc.hiroya;

public class ApspOutput {

    public static void print_int(int value) {
        System.out.print(value == ApspResolver.INF.INT_INF ? "Inf" : value);
    }

    public static void print_float(float value) {
        System.out.print(value == ApspResolver.INF.FLT_INF ? "Inf" : value);
    }

    public static void print_double(double value) {
        System.out.print(value == ApspResolver.INF.DBL_INF ? "Inf" : value);
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
