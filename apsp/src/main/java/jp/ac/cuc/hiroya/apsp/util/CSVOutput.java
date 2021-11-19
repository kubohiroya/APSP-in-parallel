package jp.ac.cuc.hiroya.apsp.util;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;

public class CSVOutput {

    static String format(float value){
        return String.format("%.2f", value);
    }
    static String format(double value){
        return String.format("%.2f", value);
    }

    static void append(StringBuffer buf, int value) {
        buf.append(value == Infinity.INT_INF ? "Inf" : value);
    }

    static void append(StringBuffer buf, float value) {
        buf.append(value == Infinity.FLT_INF ? "Inf" : format(value));
    }

    static void append(StringBuffer buf, double value) {
        buf.append(value == Infinity.DBL_INF ? "Inf" : format(value));
    }

    public static void print(int[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n]);
            for (int j = 1; j < n; j++) {
                buf.append(",");
                append(buf, matrix[i * n + j]);
            }
            System.out.println(buf);
        }
    }

    public static void print(float[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n]);
            for (int j = 1; j < n; j++) {
                buf.append(",");
                append(buf, matrix[i * n + j]);
            }
            System.out.println(buf);
        }
    }

    public static void print(double[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n]);
            for (int j = 1; j < n; j++) {
                buf.append(",\t");
                append(buf, matrix[i * n + j]);
            }
            System.out.println(buf);
        }
    }
}
