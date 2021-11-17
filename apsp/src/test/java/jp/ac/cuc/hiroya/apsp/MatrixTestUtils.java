package jp.ac.cuc.hiroya.apsp;

import jp.ac.cuc.hiroya.apsp.lib.Infinity;
import jp.ac.cuc.hiroya.apsp.util.CSVParser;
import jp.ac.cuc.hiroya.apsp.util.InfinityConverter;
import jp.ac.cuc.hiroya.apsp.util.PostcedessorNormalizer;

public class MatrixTestUtils {
    public static double calculateDistance(int from, int to, int v, double[] distances, int[] nodes, boolean verbose){

        int index = from * v + to;
        if(verbose) System.out.println(from+"発 => "+to+"行\t"+distances[index]);

        double distance = 0.0;

        while(true){
            int next = nodes[from * v + to];
            if(from == next){
                distance += distances[from * v + to];
                if(verbose) System.out.println("   "+from+"発 => "+to+"行\t最終合計距離="+distance);
                break;
            }
            distance += distances[from * v + next];
            if(next == to){
                if(verbose) System.out.println("   "+from+"発 => "+to+"行\t最終合計距離="+distance);
                break;
            }
            if(verbose) System.out.println("   "+from+"発 => "+next+"経由 (最終目的地"+to+")\t合計距離="+distance);
            from = next;
        }
        return distance;
    }

    static double[] loadAdjMatrix(String adjFilename)throws Exception{
        return InfinityConverter.convert(CSVParser.parseDoubleCSV(adjFilename), 0.0, Infinity.DBL_INF);
    }

    static double[] generateDistanceMatrix(double[] distances, int[] nodes, boolean verbose){
        int v = (int) Math.sqrt(distances.length);
        double[] ret = new double[v * v];
        for(int i = 0; i < v; i++){
            for(int j = 0; j < v; j++){
                ret[i*v + j] = MatrixTestUtils.calculateDistance(i, j, v, distances, nodes, verbose);
            }
        }
        return ret;
    }

    static double[] loadDistanceMatrix(String distanceFilename)throws Exception{
        return InfinityConverter.convert(CSVParser.parseDoubleCSV(distanceFilename), 0.0, Infinity.DBL_INF);
    }

    static int[] loadNodeMatrix(String nodeFilename)throws Exception{
        return PostcedessorNormalizer.normalize(CSVParser.parseIntCSV(nodeFilename));
    }

}
