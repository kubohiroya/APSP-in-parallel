package jp.ac.cuc.hiroya.apsp.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class CSVParser {

    public static int[] parseIntCSV(String filename) throws IOException {
        int[] ret = null;
        int p = 0;
        File file = new File(filename);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    int n = row[row.length - 1] != "" ? row.length : row.length - 1;
                    ret = new int[n * n];
                }
                for (String s : row) {
                    try {
                        ret[p++] = Integer.parseInt(s);
                    } catch (NumberFormatException ignore) {
                    }
                }
            }
        }
        return ret;
    }

    public static float[] parseFloatCSV(String filename) throws IOException {
        float[] ret = null;
        int p = 0;
        File file = new File(filename);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    int n = row[row.length - 1] != "" ? row.length : row.length - 1;
                    ret = new float[n * n];
                }
                for (String s : row) {
                    try {
                        ret[p++] = Float.parseFloat(s);
                    } catch (NumberFormatException ignore) {
                    }
                }
            }
        }
        return ret;
    }

    public static double[] parseDoubleCSV(String filename) throws IOException {
        double[] ret = null;
        int p = 0;
        File file = new File(filename);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] row = line.split(",");
                if (ret == null) {
                    int n = row[row.length - 1] != "" ? row.length : row.length - 1;
                    ret = new double[n * n];
                }
                for (String s : row) {
                    try {
                        ret[p++] = Double.parseDouble(s);
                    } catch (NumberFormatException ignore) {
                    }

                }
            }
        }
        return ret;
    }
}
