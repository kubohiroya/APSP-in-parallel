package jp.ac.cuc.hiroya.apsp.util;

public class InfinityConverter{

    public static int[] convert(int[] matrix, int targetValue, int infinityValue){
        int v = (int)Math.sqrt(matrix.length);
        for(int i = 0; i < matrix.length; i++){
            if(matrix[i] == targetValue && Math.floorDiv(i, v) != Math.floorMod(i, v)){
                matrix[i] = infinityValue;
            }
        }
        return matrix;
    }

    public static float[] convert(float[] matrix, float targetValue, float infinityValue){
        int v = (int)Math.sqrt(matrix.length);
        for(int i = 0; i < matrix.length; i++){
            if(matrix[i] == targetValue && Math.floorDiv(i, v) != Math.floorMod(i, v)){
                matrix[i] = infinityValue;
            }
        }
        return matrix;
    }

    public static double[] convert(double[] matrix, double targetValue, double infinityValue){
        int v = (int)Math.sqrt(matrix.length);
        for(int i = 0; i < matrix.length; i++){
            if(matrix[i] == targetValue && Math.floorDiv(i, v) != Math.floorMod(i, v)){
                matrix[i] = infinityValue;
            }
        }
        return matrix;
    }
}