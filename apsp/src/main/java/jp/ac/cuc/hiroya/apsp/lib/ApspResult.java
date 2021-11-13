package jp.ac.cuc.hiroya.apsp.lib;

public class ApspResult<T> {
    T output;
    int[] postdecessors;
    int numVertex;
    long elapsedTime;

    ApspResult(T output, int[] postdecessors, int numVertex, long elapsedTime) {
        this.output = output;
        this.postdecessors = postdecessors;
        this.numVertex = numVertex;
        this.elapsedTime = elapsedTime;
    }

    public T getOutput() {
        return output;
    }

    public int[] getPostdecessors() {
        return postdecessors;
    }

    public int getNumVertex() {
        return numVertex;
    }

    public long getElapsedTime() {
        return elapsedTime;
    }
}
