package jp.ac.cuc.hiroya.apsp.util;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;

public class ApspOutput {

    public static void append(StringBuffer buf, int value) {
        buf.append(value == Infinity.INT_INF ? "Inf" : value);
    }

    public static void append(StringBuffer buf, float value) {
        buf.append(value == Infinity.FLT_INF ? "Inf" : value);
    }

    public static void append(StringBuffer buf, double value) {
        buf.append(value == Infinity.DBL_INF ? "Inf" : value);
    }

    public static void print_matrix_int(int[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n]);
            for (int j = 1; j < n; j++) {
                buf.append(",");
                append(buf, matrix[i * n + j]);
            }
            System.out.println(buf.toString());
        }
    }

    public static void print_matrix_float(float[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n]);
            for (int j = 1; j < n; j++) {
                buf.append(",");
                append(buf, matrix[i * n + j]);
            }
            System.out.println(buf.toString());
        }
    }

    public static void print_matrix_double(double[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n]);
            for (int j = 1; j < n; j++) {
                buf.append(",");
                append(buf, matrix[i * n + j]);
            }
            System.out.println(buf.toString());
        }
    }
}
