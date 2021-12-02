package jp.ac.cuc.hiroya.apsp.lib;

import com.sun.jna.ptr.PointerByReference;

public class ApspResolvers {

    private static class ApspResolverIntImpl implements ApspResolver<int[]> {

        public ApspResult<int[]> resolveWithJohnson(String execEnv, int[] adjacencyMatrix){
            return resolve(execEnv, ALGORITHM.JOHNSON, adjacencyMatrix, -1);
        }

        public ApspResult<int[]> resolveWithFloydWarshall(String execEnv, int[] adjacencyMatrix){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, adjacencyMatrix, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<int[]> resolveWithFloydWarshall(String execEnv, int[] adjacencyMatrix, int numBlocks){

            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, adjacencyMatrix, numBlocks);
        }

        public ApspResult<int[]> resolve(String execEnv, String algorithm, int[] adjacencyMatrix, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference distanceMatrix = new PointerByReference();
            PointerByReference successorMatrix = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int)Math.sqrt(adjacencyMatrix.length);
            if(numVertex * numVertex != adjacencyMatrix.length){
                throw new RuntimeException("Invalid adjacencyMatrix");
            }
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.floyd_warshall_blocked_int(adjacencyMatrix, distanceMatrix, successorMatrix, numBlocks, numVertex);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.johnson_parallel_matrix_int(adjacencyMatrix, distanceMatrix, successorMatrix, numVertex);
                    break;
            }
            int[] distanceMatrixResult = distanceMatrix.getValue().getIntArray(0, numVertex * numVertex);
            int[] successorMatrixResult = successorMatrix.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.free_floyd_warshall_blocked_int(distanceMatrix, successorMatrix);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.free_johnson_parallel_matrix_int(distanceMatrix, successorMatrix);
                    break;
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<int[]>(distanceMatrixResult, successorMatrixResult, numVertex, timeEnd - timeStart);
        }

        public int[] getInfinity(String execEnv){
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            return new int[]{impl.get_infinity_int()};
        }
    }

    private static class ApspResolverFloatImpl implements ApspResolver<float[]> {

        public ApspResult<float[]> resolveWithJohnson(String execEnv, float[] adjacencyMatrix){
            return resolve(execEnv, ALGORITHM.JOHNSON, adjacencyMatrix,-1);
        }

        public ApspResult<float[]> resolveWithFloydWarshall(String execEnv, float[] adjacencyMatrix){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, adjacencyMatrix, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<float[]> resolveWithFloydWarshall(String execEnv, float[] adjacencyMatrix, int numBlocks){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, adjacencyMatrix, numBlocks);
        }

        public ApspResult<float[]> resolve(String execEnv, String algorithm, float[] adjacencyMatrix, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference distanceMatrix = new PointerByReference();
            PointerByReference successorMatrix = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int)Math.sqrt(adjacencyMatrix.length);
            if(numVertex * numVertex != adjacencyMatrix.length){
                throw new RuntimeException("Invalid adjacencyMatrix");
            }
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.floyd_warshall_blocked_float(adjacencyMatrix, distanceMatrix, successorMatrix, numBlocks, numVertex);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.johnson_parallel_matrix_float(adjacencyMatrix, distanceMatrix, successorMatrix, numVertex);
                    break;
            }
            float[] distanceMatrixResult = distanceMatrix.getValue().getFloatArray(0, numVertex * numVertex);
            int[] successorMatrixResult = successorMatrix.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.free_floyd_warshall_blocked_float(distanceMatrix, successorMatrix);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.free_johnson_parallel_matrix_float(distanceMatrix, successorMatrix);
                    break;
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<float[]>(distanceMatrixResult, successorMatrixResult, numVertex, timeEnd - timeStart);
        }

        public float[] getInfinity(String execEnv){
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            return new float[]{impl.get_infinity_float()};
        }
    }

    private static class ApspResolverDoubleImpl implements ApspResolver<double[]> {

        public ApspResult<double[]> resolveWithJohnson(String execEnv, double[] adjacencyMatrix){
            return resolve(execEnv, ALGORITHM.JOHNSON, adjacencyMatrix, -1);
        }

        public ApspResult<double[]> resolveWithFloydWarshall(String execEnv, double[] adjacencyMatrix){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, adjacencyMatrix, ALGORITHM.FLOYD_WARSHALL_BLOCK_SIZE);
        }

        public ApspResult<double[]> resolveWithFloydWarshall(String execEnv, double[] adjacencyMatrix, int numBlocks){
            return resolve(execEnv, ALGORITHM.FLOYD_WARSHALL, adjacencyMatrix, numBlocks);
        }

        public ApspResult<double[]> resolve(String execEnv, String algorithm, double[] adjacencyMatrix, int numBlocks){
            long timeStart = System.currentTimeMillis();
            PointerByReference distanceMatrix = new PointerByReference();
            PointerByReference successorMatrix = new PointerByReference();
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            int numVertex = (int)Math.sqrt(adjacencyMatrix.length);
            if(numVertex * numVertex != adjacencyMatrix.length){
                throw new RuntimeException("Invalid adjacencyMatrix");
            }
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.floyd_warshall_blocked_double(adjacencyMatrix, distanceMatrix, successorMatrix, numBlocks, numVertex);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.johnson_parallel_matrix_double(adjacencyMatrix, distanceMatrix, successorMatrix, numVertex);
                    break;
            }
            double[] distanceMatrixResult = distanceMatrix.getValue().getDoubleArray(0, numVertex * numVertex);
            int[] successorMatrixResult = successorMatrix.getValue().getIntArray(0, numVertex * numVertex);
            switch(algorithm){
                case ALGORITHM.FLOYD_WARSHALL:
                case ALGORITHM.F:
                    impl.free_floyd_warshall_blocked_double(distanceMatrix, successorMatrix);
                    break;
                case ALGORITHM.JOHNSON:
                case ALGORITHM.J:
                default:
                    impl.free_johnson_parallel_matrix_double(distanceMatrix, successorMatrix);
                    break;
            }
            long timeEnd = System.currentTimeMillis();
            return new ApspResult<double[]>(distanceMatrixResult, successorMatrixResult, numVertex, timeEnd - timeStart);
        }

        public double[] getInfinity(String execEnv){
            ApspNativeLibrary impl = ApspNativeLibraries.getImplementation(execEnv);
            return new double[]{impl.get_infinity_double()};
        }
    }

    public static ApspResolver<int[]> IntResolver = new ApspResolvers.ApspResolverIntImpl();
    public static ApspResolver<float[]> FloatResolver = new ApspResolvers.ApspResolverFloatImpl();
    public static ApspResolver<double[]> DoubleResolver = new ApspResolvers.ApspResolverDoubleImpl();

}
