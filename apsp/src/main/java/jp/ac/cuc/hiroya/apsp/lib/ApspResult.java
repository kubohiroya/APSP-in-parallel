package jp.ac.cuc.hiroya.apsp.lib;

public class ApspResult<T> {
    T distanceMatrix;
    int[] successorMatrix;
    int numVertex;
    long elapsedTime;

    ApspResult(T distanceMatrix, int[] successorMatrix, int numVertex, long elapsedTime) {
        this.distanceMatrix = distanceMatrix;
        this.successorMatrix = successorMatrix;
        this.numVertex = numVertex;
        this.elapsedTime = elapsedTime;
    }

    public T getDistanceMatrix() {
        return distanceMatrix;
    }

    public int[] getSuccessorMatrix() {
        return successorMatrix;
    }

    public int getNumVertex() {
        return numVertex;
    }

    public long getElapsedTime() {
        return elapsedTime;
    }
}
