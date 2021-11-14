package jp.ac.cuc.hiroya.apsp.util;

public class PostcedessorNormalizer {
    public static int[] normalize(int[] matrix){
        int v = (int)Math.sqrt(matrix.length);
        for(int i = 0; i < v; i++){
            matrix[i * v + i] = i;
        }
        return matrix;
    }
}
