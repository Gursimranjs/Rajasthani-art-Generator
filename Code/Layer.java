public interface Layer {
    double[] forward(double[] input);

    double[] backward(double[] gradOutput);

    void updateWeights(double learningRate);


}
