public class Discriminator {
    private Layer[] layers;
    private double[] inputGradients;

    public Discriminator(int inputSize) {
        double leakyAlpha = 0.2; // Define the negative slope for leaky ReLU

        layers = new Layer[]{
                new DenseLayer(inputSize, 512, new AdamOptimizer(0.00005, 512, inputSize)),
                new BatchNormalizationLayer(512), // Added Batch Normalization
                new DropoutLayer(0.3), // Added Dropout layer
                new ActivationLayer(ActivationFunctions.leakyReLU(leakyAlpha), ActivationFunctions.leakyReLUDerivative(leakyAlpha)), // Use parametrized Leaky ReLU
                new DenseLayer(512, 256, new AdamOptimizer(0.00005, 256, 512)),
                new BatchNormalizationLayer(256), // Added Batch Normalization
                new DropoutLayer(0.3), // Added Dropout layer
                new ActivationLayer(ActivationFunctions.leakyReLU(leakyAlpha), ActivationFunctions.leakyReLUDerivative(leakyAlpha)), // Use parametrized Leaky ReLU
                new DenseLayer(256, 1, new AdamOptimizer(0.00005, 1, 256)),
                new ActivationLayer(ActivationFunctions.sigmoid, ActivationFunctions.sigmoidDerivative) // Sigmoid activation for binary output
        };
    }

    public double[] forward(double[] input) {
        double[] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public void backward(double[] gradOutput) {
        for (int i = layers.length - 1; i >= 0; i--) {
            gradOutput = layers[i].backward(gradOutput);
        }
        inputGradients = gradOutput; // Store gradients for the input layer
    }

    public void updateWeights(double learningRate) {
        for (Layer layer : layers) {
            layer.updateWeights(learningRate); // Update weights and biases in each layer
        }
    }

    public double[] getInputGradients() {
        return inputGradients; // Used to propagate gradients to the generator
    }

    public double train(double[] input, double target) {
        // Forward pass
        double[] output = forward(input);

        // Compute loss (Least Squares Loss)
        double loss = Math.pow(output[0] - target, 2);

        // Compute gradient of loss
        double[] gradOutput = new double[1];
        gradOutput[0] = 2 * (output[0] - target); // Derivative of L2 loss

        // Backward pass
        backward(gradOutput);

        // Return loss for logging
        return loss;
    }
}
