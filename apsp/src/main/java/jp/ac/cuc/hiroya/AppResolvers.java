package jp.ac.cuc.hiroya;

import com.sun.jna.ptr.PointerByReference;

public class AppResolvers {

    static class ApspResolverIntImpl implements ApspResolver<int[]> {

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
            PointerByReference parents = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int) Math.sqrt(input.length);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.floyd_warshall_blocked_int(input, output, parents, numVertex, numBlocks);
                case ALGORITHM.JOHNSON:
                default:
                    impl.johnson_parallel_matrix_int(input, output, parents, numVertex);
            }
            int[] outputResult = output.getValue().getIntArray(0, numVertex * numVertex);
            int[] parentsResult = parents.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.free_floyd_warshall_blocked_int(output, parents);
                case ALGORITHM.JOHNSON:
                default:
                    impl.free_johnson_parallel_matrix_int(output, parents);
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<int[]>(outputResult, parentsResult,numVertex, timeEnd - timeStart);
        }
    }

    static class ApspResolverFloatImpl implements ApspResolver<float[]> {

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
            PointerByReference parents = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int) Math.sqrt(input.length);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.floyd_warshall_blocked_float(input, output, parents, numVertex, numBlocks);
                case ALGORITHM.JOHNSON:
                default:
                    impl.johnson_parallel_matrix_float(input, output, parents, numVertex);
            }
            float[] outputResult = output.getValue().getFloatArray(0, numVertex * numVertex);
            int[] parentsResult = parents.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.free_floyd_warshall_blocked_float(output, parents);
                case ALGORITHM.JOHNSON:
                default:
                    impl.free_johnson_parallel_matrix_float(output, parents);
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<float[]>(outputResult, parentsResult,numVertex, timeEnd - timeStart);
        }
    }

    static class ApspResolverDoubleImpl implements ApspResolver<double[]> {

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
            PointerByReference parents = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int) Math.sqrt(input.length);

            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.floyd_warshall_blocked_double(input, output, parents, numVertex, nuBlocks);
                case ALGORITHM.JOHNSON:
                default:
                    impl.johnson_parallel_matrix_double(input, output, parents, numVertex);
            }
            double[] outputResult = output.getValue().getDoubleArray(0, numVertex * numVertex);
            int[] parentsResult = parents.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                    impl.free_floyd_warshall_blocked_double(output, parents);
                case ALGORITHM.JOHNSON:
                default:
                    impl.free_johnson_parallel_matrix_double(output, parents);
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<double[]>(outputResult, parentsResult, numVertex,timeEnd - timeStart);
        }
    }

    public static ApspResolver<int[]> IntResolver = new AppResolvers.ApspResolverIntImpl();
    public static ApspResolver<float[]> FloatResolver = new AppResolvers.ApspResolverFloatImpl();
    public static ApspResolver<double[]> DoubleResolver = new AppResolvers.ApspResolverDoubleImpl();

}
