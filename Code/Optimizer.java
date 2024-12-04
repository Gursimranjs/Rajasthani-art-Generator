public interface Optimizer {
    void update(double[][] weights, double[][] gradWeights);
    void update(double[] biases, double[] gradBiases);
}