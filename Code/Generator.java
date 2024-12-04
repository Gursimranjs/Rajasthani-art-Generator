public class Generator {
    private Layer[] layers;

    public Generator(int noiseSize, int outputSize) {
        double leakyAlpha = 0.2; // Define the negative slope for leaky ReLU

        layers = new Layer[]{
                new DenseLayer(noiseSize, 256, new AdamOptimizer(0.0002, 256, noiseSize)),
                new BatchNormalizationLayer(256), // Batch Normalization to stabilize training
                new ActivationLayer(ActivationFunctions.leakyReLU(leakyAlpha), ActivationFunctions.leakyReLUDerivative(leakyAlpha)), // Use parametrized Leaky ReLU
                new DenseLayer(256, 512, new AdamOptimizer(0.0002, 512, 256)),
                new BatchNormalizationLayer(512), // Batch Normalization to stabilize training
                new ActivationLayer(ActivationFunctions.leakyReLU(leakyAlpha), ActivationFunctions.leakyReLUDerivative(leakyAlpha)), // Use parametrized Leaky ReLU
                new DenseLayer(512, outputSize, new AdamOptimizer(0.0002, outputSize, 512)),
                new ActivationLayer(ActivationFunctions.tanh, ActivationFunctions.tanhDerivative) // Output normalized to [-1, 1]
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
            gradOutput = layers[i].backward(gradOutput); // Propagate gradients back through the layers
        }
    }

    public void updateWeights(double learningRate) {
        for (Layer layer : layers) {
            layer.updateWeights(learningRate); // Update weights for all layers
        }
    }

    public double train(double[] noise, Discriminator discriminator) {
        // Forward pass through generator
        double[] fakeData = forward(noise);

        // Get discriminator output for fake data
        double[] discriminatorOutput = discriminator.forward(fakeData);

        // Compute loss (Generator tries to make Discriminator output 1)
        double loss = Math.pow(discriminatorOutput[0] - 1, 2);

        // Compute gradient of loss w.r.t. generator's output
        double[] gradOutput = new double[1];
        gradOutput[0] = 2 * (discriminatorOutput[0] - 1);

        // Backward pass through discriminator and generator
        discriminator.backward(gradOutput);
        backward(discriminator.getInputGradients());

        // Return loss for logging
        return loss;
    }
}
