import java.util.function.Function;

// This class represents an activation layer in a neural network, which applies a specified activation function
// to its input during the forward pass and computes gradients using the function's derivative during the backward pass.

public class ActivationLayer implements Layer {
    private Function<Double, Double> activationFunc;
    private Function<Double, Double> activationDerivative;
    private double[] input;

    public ActivationLayer(Function<Double, Double> func, Function<Double, Double> derivative) {
        this.activationFunc = func;
        this.activationDerivative = derivative;
    }

    @Override
    public double[] forward(double[] input) {
        this.input = input;
        return Matrix.applyFunction(input, activationFunc);
    }

    @Override
    public double[] backward(double[] gradOutput) {
        double[] gradInput = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            gradInput[i] = gradOutput[i] * activationDerivative.apply(input[i]);
        }
        return gradInput;
    }

    @Override
    public void updateWeights(double learningRate) {
        // No weights to update in activation layers
    }
}