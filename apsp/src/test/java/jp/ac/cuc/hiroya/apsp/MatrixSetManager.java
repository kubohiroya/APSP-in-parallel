package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.ApspResolvers;
import jp.ac.cuc.hiroya.apsp.lib.ApspResult;
import jp.ac.cuc.hiroya.apsp.lib.Infinity;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static jp.ac.cuc.hiroya.apsp.DateTimeUtil.getYYMMDDHHMM;

public class MatrixSetManager {

    static MatrixSetManager instance = new MatrixSetManager();

    public static MatrixSetManager getInstance(){
        return instance;
    }

    Map<String, MatrixSet.Double> cacheMatrixDouble = new HashMap<>();
    Map<String, MatrixSet.Float> cacheMatrixFloat = new HashMap<>();
    Map<String, MatrixSet.Int> cacheMatrixInt = new HashMap<>();
    Map<String, ApspResult<double[]>> cacheResultDouble = new HashMap<>();
    Map<String, ApspResult<float[]>> cacheResultFloat = new HashMap<>();
    Map<String, ApspResult<int[]>> cacheResultInt = new HashMap<>();

    synchronized MatrixSet.Double getMatrixSetDouble(String adjacencyFilename, String distanceFilename, String successorFilename) throws IOException,NumberFormatException{
        String key = adjacencyFilename;
        MatrixSet.Double set = cacheMatrixDouble.get(key);
        if (set == null) {
            System.out.println(getYYMMDDHHMM() + " prepare 1: load csv key="+key);
            System.out.println("   adj: " + adjacencyFilename);
            double[] adjacencyMatrix = MatrixUtil.getAdjacencyMatrix(adjacencyFilename, Infinity.DBL_INF);
            System.out.println("   dst: " + distanceFilename);
            double[] distanceMatrix = distanceFilename!=null? MatrixUtil.loadCsvFileDouble(distanceFilename):null;
            System.out.println("   suc: " + successorFilename);
            int[] successorMatrix = successorFilename!=null? MatrixUtil.loadSuccessorMatrix(successorFilename):null;
            set = new MatrixSet.Double(adjacencyMatrix, distanceMatrix, successorMatrix);
            cacheMatrixDouble.put(key, set);
        }else{
            System.out.println(getYYMMDDHHMM() + " prepare 1: use cached csv key="+key);
        }
        return set;
    }

    synchronized MatrixSet.Float getMatrixSetFloat(String adjacencyFilename, String distanceFilename, String successorFilename) throws IOException,NumberFormatException{
        String key = adjacencyFilename;
        MatrixSet.Float set = cacheMatrixFloat.get(key);
        if (set == null) {
            System.out.println(getYYMMDDHHMM() + " prepare 1: load csv ");
            System.out.println("   adj: " + adjacencyFilename);
            float[] adjacencyMatrix = MatrixUtil.getAdjacencyMatrix(adjacencyFilename, Infinity.FLT_INF);
            System.out.println("   dst: " + distanceFilename);
            float[] distanceMatrix = distanceFilename!=null? MatrixUtil.loadCsvFileFloat(distanceFilename):null;
            System.out.println("   suc: " + successorFilename);
            int[] successorMatrix = successorFilename!=null? MatrixUtil.loadSuccessorMatrix(successorFilename):null;
            set = new MatrixSet.Float(adjacencyMatrix, distanceMatrix, successorMatrix);
            cacheMatrixFloat.put(key, set);
        }
        return set;
    }

    synchronized MatrixSet.Int getMatrixSetInt(String adjacencyFilename, String distanceFilename, String successorFilename) throws IOException,NumberFormatException{
        String key = adjacencyFilename;
        MatrixSet.Int set = cacheMatrixInt.get(key);
        if (set == null) {
            System.out.println(getYYMMDDHHMM() + " prepare 1: load csv ");
            System.out.println("   adj: " + adjacencyFilename);
            int[] adjacencyMatrix = MatrixUtil.getAdjacencyMatrix(adjacencyFilename, Infinity.INT_INF);
            System.out.println("   dst: " + distanceFilename);
            int[] distanceMatrix = distanceFilename!=null? MatrixUtil.loadCsvFileInt(distanceFilename):null;
            System.out.println("   suc: " + successorFilename);
            int[] successorMatrix = successorFilename!=null? MatrixUtil.loadSuccessorMatrix(successorFilename):null;
            set = new MatrixSet.Int(adjacencyMatrix, distanceMatrix, successorMatrix);
            cacheMatrixInt.put(key, set);
        }
        return set;
    }

    synchronized ApspResult<double[]> getApspResultDouble(String adjacencyFilename, String execEnv, String algorithm) throws IOException{
        String key = adjacencyFilename+"\t"+execEnv + ":" + algorithm;
        ApspResult<double[]> result = cacheResultDouble.get(key);
        if (result == null) {
            double[] adjacencyMatrix = getMatrixSetDouble(adjacencyFilename,
                    null, null).getAdjacencyMatrix();
            System.out.println(getYYMMDDHHMM()+" prepare 2: resolve all-pairs-shortest-paths " + execEnv + "-" + algorithm);
            result = ApspResolvers.DoubleResolver.resolve(execEnv, algorithm,
                    adjacencyMatrix, 64);
            cacheResultDouble.put(key, result);
        }
        return result;
    }

    synchronized ApspResult<float[]> getApspResultFloat(String adjacencyFilename, String execEnv, String algorithm) throws IOException{
        String key = adjacencyFilename+"\t"+execEnv + ":" + algorithm;
        ApspResult<float[]> result = cacheResultFloat.get(key);
        if (result == null) {
            float[] adjacencyMatrix = getMatrixSetFloat(adjacencyFilename,
                    null, null).getAdjacencyMatrix();
            System.out.println(getYYMMDDHHMM()+" step 2: resolve all-pairs-shortest-paths " + execEnv + "-" + algorithm);
            result = ApspResolvers.FloatResolver.resolve(execEnv, algorithm,
                    adjacencyMatrix, 64);
            cacheResultFloat.put(key, result);
        }
        return result;
    }

    synchronized ApspResult<int[]> getApspResultInt(String adjacencyFilename, String execEnv, String algorithm) throws IOException{
        String key = adjacencyFilename+"\t"+execEnv + ":" + algorithm;
        ApspResult<int[]> result = cacheResultInt.get(key);
        if (result == null) {
            int[] adjacencyMatrix = getMatrixSetInt(adjacencyFilename,
                    null, null).getAdjacencyMatrix();
            System.out.println(getYYMMDDHHMM()+" step 2: resolve all-pairs-shortest-paths " + execEnv + "-" + algorithm);
            result = ApspResolvers.IntResolver.resolve(execEnv, algorithm,
                    adjacencyMatrix, 64);
            cacheResultInt.put(key, result);
        }
        return result;
    }

}
