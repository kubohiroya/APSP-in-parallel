package jp.ac.cuc.hiroya.apsp.lib;

import com.sun.jna.ptr.PointerByReference;

public class ApspResolvers {

    private ApspResolvers(){}

    private static class ApspResolverIntImpl implements ApspResolver<int[]> {

        public ApspResult<int[]> resolveWithJohnson(String execEnv, int[] input, int numVertex){
            return resolve(execEnv, ALGORITHM.JOHNSON, input, numVertex, -1);
        }

        public ApspResult<int[]> resolveWithFloydWarshall(String execEnv, int[] input, int numVertex){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numVertex, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<int[]> resolveWithFloydWarshall(String execEnv, int[] input, int numVertex, int numBlocks){

            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numVertex, numBlocks);
        }

        public ApspResult<int[]> resolve(String execEnv, String algorithm, int[] input, int numVertex, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference output = new PointerByReference();
            PointerByReference postdecessors = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.floyd_warshall_blocked_int(input, output, postdecessors, numVertex, numBlocks);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.johnson_parallel_matrix_int(input, output, postdecessors, numVertex);
                    break;
            }
            int[] outputResult = output.getValue().getIntArray(0, numVertex * numVertex);
            int[] postdecessorsResult = postdecessors.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.free_floyd_warshall_blocked_int(output, postdecessors);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.free_johnson_parallel_matrix_int(output, postdecessors);
                    break;
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<int[]>(outputResult, postdecessorsResult, numVertex, timeEnd - timeStart);
        }
    }

    private static class ApspResolverFloatImpl implements ApspResolver<float[]> {

        public ApspResult<float[]> resolveWithJohnson(String execEnv, float[] input, int numVertex){
            return resolve(execEnv, ALGORITHM.JOHNSON, input, numVertex, -1);
        }

        public ApspResult<float[]> resolveWithFloydWarshall(String execEnv, float[] input, int numVertex){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numVertex, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<float[]> resolveWithFloydWarshall(String execEnv, float[] input, int numVertex, int numBlocks){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numVertex, numBlocks);
        }

        public ApspResult<float[]> resolve(String execEnv, String algorithm, float[] input, int numVertex, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference output = new PointerByReference();
            PointerByReference postdecessors = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.floyd_warshall_blocked_float(input, output, postdecessors, numVertex, numBlocks);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.johnson_parallel_matrix_float(input, output, postdecessors, numVertex);
                    break;
            }
            float[] outputResult = output.getValue().getFloatArray(0, numVertex * numVertex);
            int[] postdecessorsResult = postdecessors.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.free_floyd_warshall_blocked_float(output, postdecessors);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.free_johnson_parallel_matrix_float(output, postdecessors);
                    break;
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<float[]>(outputResult, postdecessorsResult, numVertex, timeEnd - timeStart);
        }
    }

    private static class ApspResolverDoubleImpl implements ApspResolver<double[]> {

        public ApspResult<double[]> resolveWithJohnson(String execEnv, double[] input, int numVertex){
            return resolve(execEnv, ALGORITHM.JOHNSON, input, numVertex, -1);
        }

        public ApspResult<double[]> resolveWithFloydWarshall(String execEnv, double[] input, int numVertex){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numVertex, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<double[]> resolveWithFloydWarshall(String execEnv, double[] input, int numVertex, int numBlocks){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, input, numVertex, numBlocks);
        }

        public ApspResult<double[]> resolve(String execEnv, String algorithm, double[] input, int numVertex, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference output = new PointerByReference();
            PointerByReference postdecessors = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.floyd_warshall_blocked_double(input, output, postdecessors, numVertex, numBlocks);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.johnson_parallel_matrix_double(input, output, postdecessors, numVertex);
                    break;
            }
            double[] outputResult = output.getValue().getDoubleArray(0, numVertex * numVertex);
            int[] postdecessorsResult = postdecessors.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.free_floyd_warshall_blocked_double(output, postdecessors);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.free_johnson_parallel_matrix_double(output, postdecessors);
                    break;
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<double[]>(outputResult, postdecessorsResult, numVertex, timeEnd - timeStart);
        }
    }

    public static ApspResolver<int[]> IntResolver = new ApspResolvers.ApspResolverIntImpl();
    public static ApspResolver<float[]> FloatResolver = new ApspResolvers.ApspResolverFloatImpl();
    public static ApspResolver<double[]> DoubleResolver = new ApspResolvers.ApspResolverDoubleImpl();

}
