package jp.ac.cuc.hiroya.apsp.util;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;

import static jp.ac.cuc.hiroya.apsp.util.ColorSeq.end;

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

    static void append(StringBuffer buf, int value, String color) {
        if(color != null){
            buf.append(color);
        }
        buf.append(value == Infinity.INT_INF ? "Inf" : value);
        if(color != null){
            buf.append(end);
        }
    }

    static void append(StringBuffer buf, float value, String color) {
        if(color != null){
            buf.append(color);
        }
        buf.append(value == Infinity.FLT_INF ? "Inf" : format(value));
        if(color != null){
            buf.append(end);
        }
    }

    static void append(StringBuffer buf, double value, String color) {
        if(color != null){
            buf.append(color);
        }
        buf.append(value == Infinity.DBL_INF ? "Inf" : format(value));
        if(color != null){
            buf.append(end);
        }
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

    public interface MatrixContext{
        boolean check(int i, int y, double value);
    }

    public static void print(double[] matrix, MatrixContext ctx, String color, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n], ctx.check(i, 0, matrix[i*n])?color:null);
            for (int j = 1; j < n; j++) {
                buf.append(",\t");
                append(buf, matrix[i * n + j], ctx.check(i, j, matrix[i*n])?color:null);
            }
            System.out.println(buf);
        }
    }

    public static void print(int[] matrix, MatrixContext ctx, String color, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n], ctx.check(i, 0, matrix[i*n])?color:null);
            for (int j = 1; j < n; j++) {
                buf.append(",\t");
                append(buf, matrix[i * n + j], ctx.check(i, j, matrix[i*n])?color:null);
            }
            System.out.println(buf);
        }
    }

    public static void printDiff(double[] matrix, boolean[] diff, String color, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n], diff[i * n]?color:null);
            for (int j = 1; j < n; j++) {
                buf.append(",\t");
                append(buf, matrix[i * n + j], diff[i * n + j]?color:null);
            }
            System.out.println(buf);
        }
    }

    public static void printDiff(double[] matrix1, double[] matrix2, String borderColor, String color1, String color2, int n) {
        boolean[] diff = new boolean[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int index = i * n + j;
                if(Math.abs(matrix1[index] - matrix2[index])>0.0001){
                    diff[index] = true;
                }
            }
        }
        System.out.println(borderColor+"--------------------"+end);
        printDiff(matrix1, diff, color1, n);
        System.out.println(borderColor+"--------------------"+end);
        printDiff(matrix2, diff, color2, n);
        System.out.println(borderColor+"--------------------"+end);
    }

    public static void printDiff(int[] matrix, boolean[] diff, String color, int n) {
        for (int i = 0; i < n; i++) {
            StringBuffer buf = new StringBuffer();
            append(buf, matrix[i * n], diff[i * n]?color:null);
            for (int j = 1; j < n; j++) {
                buf.append(",\t");
                append(buf, matrix[i * n + j], diff[i * n + j]?color:null);
            }
            System.out.println(buf);
        }
    }

    public static void printDiff(int[] matrix1, int[] matrix2, String borderColor, String color1, String color2, int n) {
        boolean[] diff = new boolean[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int index = i * n + j;
                if(Math.abs(matrix1[index] - matrix2[index])>0.0001){
                    diff[index] = true;
                }
            }
        }
        System.out.println(borderColor+"--------------------"+end);
        printDiff(matrix1, diff, color1, n);
        System.out.println(borderColor+"--------------------"+end);
        printDiff(matrix2, diff, color2, n);
        System.out.println(borderColor+"--------------------"+end);
    }
}
