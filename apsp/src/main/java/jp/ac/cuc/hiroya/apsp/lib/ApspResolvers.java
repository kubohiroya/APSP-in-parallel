package jp.ac.cuc.hiroya.apsp.lib;

import com.sun.jna.ptr.PointerByReference;

public class ApspResolvers {

    private ApspResolvers(){}

    private static class ApspResolverIntImpl implements ApspResolver<int[]> {

        public ApspResult<int[]> resolveWithJohnson(String execEnv, int[] input){
            return resolve(execEnv, ALGORITHM.JOHNSON, input, -1);
        }

        public ApspResult<int[]> resolveWithFloydWarshall(String execEnv, int[] input){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<int[]> resolveWithFloydWarshall(String execEnv, int[] input, int numBlocks){

            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numBlocks);
        }

        public ApspResult<int[]> resolve(String execEnv, String algorithm, int[] input, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference output = new PointerByReference();
            PointerByReference predecessors = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int) Math.sqrt(input.length);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.floyd_warshall_blocked_int(input, output, predecessors, numVertex, numBlocks);
                case ALGORITHM.JOHNSON:
                default:
                    impl.johnson_parallel_matrix_int(input, output, predecessors, numVertex);
            }
            int[] outputResult = output.getValue().getIntArray(0, numVertex * numVertex);
            int[] predecessorsResult = predecessors.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.free_floyd_warshall_blocked_int(output, predecessors);
                case ALGORITHM.JOHNSON:
                default:
                    impl.free_johnson_parallel_matrix_int(output, predecessors);
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<int[]>(outputResult, predecessorsResult,numVertex, timeEnd - timeStart);
        }
    }

    private static class ApspResolverFloatImpl implements ApspResolver<float[]> {

        public ApspResult<float[]> resolveWithJohnson(String execEnv, float[] input){
            return resolve(execEnv, ALGORITHM.JOHNSON, input, -1);
        }

        public ApspResult<float[]> resolveWithFloydWarshall(String execEnv, float[] input){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<float[]> resolveWithFloydWarshall(String execEnv, float[] input, int numBlocks){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numBlocks);
        }

        public ApspResult<float[]> resolve(String execEnv, String algorithm, float[] input, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference output = new PointerByReference();
            PointerByReference predecessors = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int) Math.sqrt(input.length);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.floyd_warshall_blocked_float(input, output, predecessors, numVertex, numBlocks);
                case ALGORITHM.JOHNSON:
                default:
                    impl.johnson_parallel_matrix_float(input, output, predecessors, numVertex);
            }
            float[] outputResult = output.getValue().getFloatArray(0, numVertex * numVertex);
            int[] predecessorsResult = predecessors.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.free_floyd_warshall_blocked_float(output, predecessors);
                case ALGORITHM.JOHNSON:
                default:
                    impl.free_johnson_parallel_matrix_float(output, predecessors);
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<float[]>(outputResult, predecessorsResult,numVertex, timeEnd - timeStart);
        }
    }

    private static class ApspResolverDoubleImpl implements ApspResolver<double[]> {

        public ApspResult<double[]> resolveWithJohnson(String execEnv, double[] input){
            return resolve(execEnv, ALGORITHM.JOHNSON, input, -1);
        }

        public ApspResult<double[]> resolveWithFloydWarshall(String execEnv, double[] input){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<double[]> resolveWithFloydWarshall(String execEnv, double[] input, int numBlocks){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numBlocks);
        }

        public ApspResult<double[]> resolve(String execEnv, String algorithm, double[] input, int nuBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference output = new PointerByReference();
            PointerByReference predecessors = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int) Math.sqrt(input.length);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.floyd_warshall_blocked_double(input, output, predecessors, numVertex, nuBlocks);
                case ALGORITHM.JOHNSON:
                default:
                    impl.johnson_parallel_matrix_double(input, output, predecessors, numVertex);
            }
            double[] outputResult = output.getValue().getDoubleArray(0, numVertex * numVertex);
            int[] predecessorsResult = predecessors.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.free_floyd_warshall_blocked_double(output, predecessors);
                case ALGORITHM.JOHNSON:
                default:
                    impl.free_johnson_parallel_matrix_double(output, predecessors);
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<double[]>(outputResult, predecessorsResult, numVertex,timeEnd - timeStart);
        }
    }

    public static ApspResolver<int[]> IntResolver = new ApspResolvers.ApspResolverIntImpl();
    public static ApspResolver<float[]> FloatResolver = new ApspResolvers.ApspResolverFloatImpl();
    public static ApspResolver<double[]> DoubleResolver = new ApspResolvers.ApspResolverDoubleImpl();

}
