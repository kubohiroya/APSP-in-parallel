package jp.ac.cuc.hiroya.apsp.util;

public class InfinityConverter{

    public static int[] convert(int[] matrix, int targetValue, int infinityValue){
        int v = (int)Math.sqrt(matrix.length);
        for(int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                if (i == j) {
                    matrix[i*v+j] = 0;
                } else if (matrix[i*v+j] == targetValue) {
                    matrix[i*v+j] = infinityValue;
                }
            }
        }
        return matrix;
    }

    public static float[] convert(float[] matrix, float targetValue, float infinityValue){
        int v = (int)Math.sqrt(matrix.length);
        for(int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                if (i == j) {
                    matrix[i*v+j] = 0.0f;
                } else if (matrix[i*v+j] == targetValue) {
                    matrix[i*v+j] = infinityValue;
                }
            }
        }
        return matrix;
    }

    public static double[] convert(double[] matrix, double targetValue, double infinityValue){
        int v = (int)Math.sqrt(matrix.length);
        for(int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                if (i == j) {
                    matrix[i*v+j] = 0.0;
                } else if (matrix[i*v+j] == targetValue) {
                    matrix[i*v+j] = infinityValue;
                }
            }
        }
        return matrix;
    }
}