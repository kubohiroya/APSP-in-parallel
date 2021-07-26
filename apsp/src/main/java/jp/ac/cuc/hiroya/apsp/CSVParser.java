package jp.ac.cuc.hiroya.apsp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class CSVParser {

    public static int[] parseCSV_int(String filename) throws IOException {
        int[] ret = null;
        int p = 0;
        File file = new File(filename);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    int n = row.length;
                    ret = new int[n * n];
                }
                for (String s : row) {
                    ret[p++] = (int) Float.parseFloat(s);
                }
            }
        }
        return ret;
    }

    public static float[] parseCSV_float(String filename) throws IOException {
        float[] ret = null;
        int p = 0;
        File file = new File(filename);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    int n = row.length;
                    ret = new float[n * n];
                }
                for (String s : row) {
                    ret[p++] = Float.parseFloat(s);
                }
            }
        }
        return ret;
    }

    public static double[] parseCSV_double(String filename) throws IOException {
        double[] ret = null;
        int p = 0;
        File file = new File(filename);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    int n = row.length;
                    ret = new double[n * n];
                }
                for (String s : row) {
                    ret[p++] = Double.parseDouble(s);
                }
            }
        }
        return ret;
    }
}
