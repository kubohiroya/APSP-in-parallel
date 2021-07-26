package jp.ac.cuc.hiroya.apsp;

import com.sun.jna.Native;

class ApspNativeLibraries {
    public interface ApspSeq extends ApspNativeLibrary {
        ApspSeq INSTANCE = Native.load("apsp-seq", ApspSeq.class);
    }

    public interface ApspSeqIspc extends ApspNativeLibrary {
        ApspSeqIspc INSTANCE = Native.load("apsp-seq-ispc", ApspSeqIspc.class);
    }

    public interface ApspOmp extends ApspNativeLibrary {
        ApspOmp INSTANCE = Native.load("apsp-omp", ApspOmp.class);
    }

    public interface ApspOmpIspc extends ApspNativeLibrary {
        ApspOmpIspc INSTANCE = Native.load("apsp-omp-ispc", ApspOmpIspc.class);
    }

    /*
    public static class ApspCuda extends ApspNativeLibrary {
        ApspCuda INSTANCE = Native.load("apsp-cuda", ApspCuda.class);
    }
     */

    public static ApspNativeLibrary getImplementation(String execEnv) {
        switch (execEnv) {
            case ApspResolver.EXEC_ENV.SEQ:
                return ApspSeq.INSTANCE;
            case ApspResolver.EXEC_ENV.SEQ_ISPC:
                return ApspSeqIspc.INSTANCE;
            case ApspResolver.EXEC_ENV.OMP_ISPC:
                return ApspOmpIspc.INSTANCE;
            case ApspResolver.EXEC_ENV.OMP:
            default:
                return ApspOmp.INSTANCE;
        }
    }
}
