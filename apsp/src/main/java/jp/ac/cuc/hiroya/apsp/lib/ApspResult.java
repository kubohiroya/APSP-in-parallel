package jp.ac.cuc.hiroya.apsp.lib;

public class ApspResult<T> {
    T output;
    int[] predecessors;
    int numVertex;
    long elapsedTime;

    ApspResult(T output, int[] predecessors, int numVertex, long elapsedTime) {
        this.output = output;
        this.predecessors = predecessors;
        this.numVertex = numVertex;
        this.elapsedTime = elapsedTime;
    }

    public T getOutput() {
        return output;
    }

    public int[] getPredecessors() {
        return predecessors;
    }

    public int getNumVertex() {
        return numVertex;
    }

    public long getElapsedTime() {
        return elapsedTime;
    }
}
