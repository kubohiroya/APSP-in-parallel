package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResult;

public class MatrixSet {
    static abstract class Base<T>{
        int numVertex;
        int[] successorMatrix;
        ApspResult<T> apspResult;
        Base(int numElements){
            this.numVertex = (int)Math.sqrt(numElements);;
            if(this.numVertex * this.numVertex != numElements){
                throw new RuntimeException("Invalid numElements:"+numElements);
            }
        }
        Base(int numElements, int[] successorMatrix){
            this(numElements);
            this.successorMatrix = successorMatrix;
        }
        int getNumVertex(){
            return this.numVertex;
        }
        int[] getSuccessorMatrix(){
            return this.successorMatrix;
        }
        public void setApspResult(ApspResult<T> apspResult){
            this.apspResult = apspResult;
        }
        public ApspResult<T>getApspResult(){
            return this.apspResult;
        }
    }
    public static class Double extends Base<double[]>{
        double[] adjacencyMatrix;
        double[] distanceMatrix;
        public Double(double[] adjacencyMatrix){
            super(adjacencyMatrix.length);
            this.adjacencyMatrix = adjacencyMatrix;
        }
        public Double(double[] adjacencyMatrix, double[] distanceMatrix, int[] successorMatrix){
            super(adjacencyMatrix.length, successorMatrix);
            this.adjacencyMatrix = adjacencyMatrix;
            this.distanceMatrix = distanceMatrix;
        }
        public double[] getAdjacencyMatrix(){
            return this.adjacencyMatrix;
        }
        public double[] getDistanceMatrix(){
            return this.distanceMatrix;
        }

    }
    public static class Float extends Base<float[]>{
        float[] adjacencyMatrix;
        float[] distanceMatrix;
        ApspResult<float[]> apspResult;
        public Float(float[] adjacencyMatrix){
            super(adjacencyMatrix.length);
            this.adjacencyMatrix = adjacencyMatrix;
        }
        public Float(float[] adjacencyMatrix, float[] distanceMatrix, int[] successorMatrix){
            super(adjacencyMatrix.length, successorMatrix);
            this.adjacencyMatrix = adjacencyMatrix;
            this.distanceMatrix = distanceMatrix;
        }
        float[] getAdjacencyMatrix(){
            return this.adjacencyMatrix;
        }
        float[] getDistanceMatrix(){
            return this.distanceMatrix;
        }
    }
    public static class Int extends Base<int[]>{
        int[] adjacencyMatrix;
        int[] distanceMatrix;
        ApspResult<int[]> apspResult;
        public Int(int[] adjacencyMatrix){
            super(adjacencyMatrix.length);
            this.adjacencyMatrix = adjacencyMatrix;
        }
        public Int(int[] adjacencyMatrix, int[] distanceMatrix, int[] successorMatrix){
            super(adjacencyMatrix.length, successorMatrix);
            this.adjacencyMatrix = adjacencyMatrix;
            this.distanceMatrix = distanceMatrix;
        }
        public int[] getAdjacencyMatrix(){
            return this.adjacencyMatrix;
        }
        public int[] getDistanceMatrix(){
            return this.distanceMatrix;
        }
    }
}

