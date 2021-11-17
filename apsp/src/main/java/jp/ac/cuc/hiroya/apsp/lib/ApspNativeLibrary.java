package jp.ac.cuc.hiroya.apsp.lib;

import com.sun.jna.Library;
import com.sun.jna.ptr.PointerByReference;

interface ApspNativeLibrary extends Library {
    int getIntegerInfinity();
    float getFloatInfinity();
    double getDoubleInfinity();

    void floyd_warshall_blocked_int(int[] adjacencyMatrix, PointerByReference distanceMatrix, PointerByReference predecessors,
                                    int n, int b);

    void free_floyd_warshall_blocked_int(PointerByReference distanceMatrix, PointerByReference predecessors);

    void floyd_warshall_blocked_float(float[] adjacencyMatrix, PointerByReference distanceMatrix, PointerByReference predecessors,
                                      int n, int b);

    void free_floyd_warshall_blocked_float(PointerByReference distanceMatrix, PointerByReference predecessors);

    void floyd_warshall_blocked_double(double[] adjacencyMatrix, PointerByReference distanceMatrix, PointerByReference predecessors,
                                       int n, int b);

    void free_floyd_warshall_blocked_double(PointerByReference distanceMatrix, PointerByReference predecessors);

    void johnson_parallel_matrix_int(int[] adjacencyMatrix, PointerByReference distanceMatrix, PointerByReference predecessors,
                                     int n);

    void free_johnson_parallel_matrix_int(PointerByReference distanceMatrix, PointerByReference predecessors);

    void johnson_parallel_matrix_float(float[] adjacencyMatrix, PointerByReference distanceMatrix, PointerByReference predecessors,
                                       int n);

    void free_johnson_parallel_matrix_float(PointerByReference distanceMatrix, PointerByReference predecessors);

    void johnson_parallel_matrix_double(double[] adjacencyMatrix, PointerByReference distanceMatrix,
                                        PointerByReference predecessors, int n);

    void free_johnson_parallel_matrix_double(PointerByReference distanceMatrix, PointerByReference predecessors);
}
