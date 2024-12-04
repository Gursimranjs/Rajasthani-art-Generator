import java.util.function.Function;


// This utility class provides common matrix and vector operations needed for neural network computations.
// It includes methods for applying functions element-wise, matrix-vector multiplication, vector addition,
// subtraction, scalar multiplication, and dot product calculation.

public class Matrix {


    public static double[] applyFunction(double[] input, Function<Double, Double> func) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = func.apply(input[i]);
        }
        return result;
    }



    public static double[] multiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            double sum = 0;
            double[] row = matrix[i];
            for (int j = 0; j < cols; j++) {
                sum += row[j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    public static double[] add(double[] vector1, double[] vector2) {
        double[] result = new double[vector1.length];
        for (int i = 0; i < vector1.length; i++) {
            result[i] = vector1[i] + vector2[i];
        }
        return result;
    }

    public static double[] subtract(double[] vector1, double[] vector2) {
        double[] result = new double[vector1.length];
        for (int i = 0; i < vector1.length; i++) {
            result[i] = vector1[i] - vector2[i];
        }
        return result;
    }

    public static double[] scalarMultiply(double[] vector, double scalar) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * scalar;
        }
        return result;
    }

    public static double dotProduct(double[] vector1, double[] vector2) {
        double sum = 0;
        for (int i = 0; i < vector1.length; i++) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }
}