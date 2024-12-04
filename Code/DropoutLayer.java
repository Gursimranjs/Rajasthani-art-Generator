import java.util.Random;

// This class implements a dropout layer for a neural network. Dropout is a regularization technique
// that randomly disables a fraction of neurons during training to prevent overfitting. The layer
// applies a dropout mask during the forward pass and propagates gradients only for active neurons
// during the backward pass.

public class DropoutLayer implements Layer {
    private double dropoutRate;
    private boolean[] mask;

    public DropoutLayer(double dropoutRate) {
        if (dropoutRate < 0.0 || dropoutRate > 1.0) {
            throw new IllegalArgumentException("Dropout rate must be between 0.0 and 1.0");
        }
        this.dropoutRate = dropoutRate;
    }

    @Override
    public double[] forward(double[] input) {
        Random rand = new Random();
        mask = new boolean[input.length];
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            mask[i] = rand.nextDouble() > dropoutRate;
            output[i] = mask[i] ? input[i] : 0.0;
        }
        return output;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[gradOutput.length];
        for (int i = 0; i < gradOutput.length; i++) {
            gradInput[i] = mask[i] ? gradOutput[i] : 0.0;
        }
        return gradInput;
    }

    @Override
    public void updateWeights(double learningRate) {
        // No weights to update in Dropout layer
    }
}
