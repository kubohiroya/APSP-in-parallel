package jp.ac.cuc.hiroya;

import com.sun.jna.Library;
import com.sun.jna.ptr.PointerByReference;

interface ApspNativeLibrary extends Library {
    void floyd_warshall_blocked_int(int[] input, PointerByReference output, PointerByReference parents,
                                    int n, int b);

    void free_floyd_warshall_blocked_int(PointerByReference output, PointerByReference parents);

    void floyd_warshall_blocked_float(float[] input, PointerByReference output, PointerByReference parents,
                                      int n, int b);

    void free_floyd_warshall_blocked_float(PointerByReference output, PointerByReference parents);

    void floyd_warshall_blocked_double(double[] input, PointerByReference output, PointerByReference parents,
                                       int n, int b);

    void free_floyd_warshall_blocked_double(PointerByReference output, PointerByReference parents);

    void johnson_parallel_matrix_int(int[] input, PointerByReference output, PointerByReference parents,
                                     int n);

    void free_johnson_parallel_matrix_int(PointerByReference output, PointerByReference parents);

    void johnson_parallel_matrix_float(float[] input, PointerByReference output, PointerByReference parents,
                                       int n);

    void free_johnson_parallel_matrix_float(PointerByReference output, PointerByReference parents);

    void johnson_parallel_matrix_double(double[] input, PointerByReference output,
                                        PointerByReference parents, int n);

    void free_johnson_parallel_matrix_double(PointerByReference output, PointerByReference parents);
}
