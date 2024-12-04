import java.util.Random;

// This class implements a fully connected (dense) layer for a neural network. It includes support for dropout
// regularization, backpropagation, and weight updates using a specified optimizer. The weights are initialized
// using Xavier/Glorot initialization for improved training stability.

public class DenseLayer implements Layer {
    private double[][] weights;
    private double[] biases;
    private double[] input;
    private double[] output;
    private double[][] gradWeights;
    private double[] gradBiases;
    private Optimizer optimizer;
    private double dropoutRate;
    private boolean[] dropoutMask;

    public DenseLayer(int inputSize, int outputSize, Optimizer optimizer) {
        this(inputSize, outputSize, optimizer, 0.0); // Default dropout rate is 0
    }

    public DenseLayer(int inputSize, int outputSize, Optimizer optimizer, double dropoutRate) {
        this.optimizer = optimizer;
        this.dropoutRate = dropoutRate;
        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];
        gradWeights = new double[outputSize][inputSize];
        gradBiases = new double[outputSize];
        dropoutMask = new boolean[outputSize];

        Random rand = new Random();
        // Xavier/Glorot Initialization for better training dynamics
        double std = Math.sqrt(2.0 / (inputSize + outputSize));
        for (int i = 0; i < outputSize; i++) {
            biases[i] = 0;
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = rand.nextGaussian() * std;
            }
        }
    }

    @Override
    public double[] forward(double[] input) {
        this.input = input;

        // Forward pass through weights and biases
        output = Matrix.add(Matrix.multiply(weights, input), biases);

        // Apply dropout if rate > 0 (only during training)
        if (dropoutRate > 0) {
            Random rand = new Random();
            for (int i = 0; i < output.length; i++) {
                dropoutMask[i] = rand.nextDouble() > dropoutRate;
                output[i] = dropoutMask[i] ? output[i] : 0.0;
            }
        }

        return output;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        int inputSize = weights[0].length;
        int outputSize = weights.length;

        // Backpropagate through dropout mask if applicable
        if (dropoutRate > 0) {
            for (int i = 0; i < gradOutput.length; i++) {
                gradOutput[i] = dropoutMask[i] ? gradOutput[i] : 0.0;
            }
        }

        // Compute gradients with respect to weights and biases
        for (int i = 0; i < outputSize; i++) {
            gradBiases[i] = gradOutput[i];
            for (int j = 0; j < inputSize; j++) {
                gradWeights[i][j] = gradOutput[i] * input[j];
            }
        }

        // Compute gradient with respect to input
        double[] gradInput = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            double sum = 0;
            for (int j = 0; j < outputSize; j++) {
                sum += weights[j][i] * gradOutput[j];
            }
            gradInput[i] = sum;
        }
        return gradInput;
    }

    @Override
    public void updateWeights(double learningRate) {
        // Gradient Clipping
        double threshold = 1.0; // This threshold can be made configurable if needed
        for (int i = 0; i < gradWeights.length; i++) {
            for (int j = 0; j < gradWeights[0].length; j++) {
                if (gradWeights[i][j] > threshold) gradWeights[i][j] = threshold;
                else if (gradWeights[i][j] < -threshold) gradWeights[i][j] = -threshold;
            }
        }
        for (int i = 0; i < gradBiases.length; i++) {
            if (gradBiases[i] > threshold) gradBiases[i] = threshold;
            else if (gradBiases[i] < -threshold) gradBiases[i] = -threshold;
        }

        // Update weights and biases using the optimizer
        optimizer.update(weights, gradWeights);
        optimizer.update(biases, gradBiases);
    }
}
