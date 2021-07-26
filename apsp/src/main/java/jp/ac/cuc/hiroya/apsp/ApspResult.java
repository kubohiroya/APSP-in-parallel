package jp.ac.cuc.hiroya.apsp;

public class ApspResult<T> {
    T output;
    int[] parents;
    int numVertex;
    long elapsedTime;

    ApspResult(T output, int[] parents, int numVertex, long elapsedTime) {
        this.output = output;
        this.parents = parents;
        this.numVertex = numVertex;
        this.elapsedTime = elapsedTime;
    }

    public T getOutput() {
        return output;
    }

    public int[] getParents() {
        return parents;
    }

    public int getNumVertex() {
        return numVertex;
    }

    public long getElapsedTime() {
        return elapsedTime;
    }
}
