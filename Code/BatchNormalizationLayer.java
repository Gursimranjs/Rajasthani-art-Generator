import java.util.Arrays;

// This class implements a batch normalization layer for a neural network.
// It normalizes the input data during the forward pass to stabilize training and
// updates scale (gamma) and shift (beta) parameters during the backward pass.


public class BatchNormalizationLayer implements Layer {
    private double[] gamma;
    private double[] beta;
    private double[] input;
    private double[] normalizedInput;
    private double[] mean;
    private double[] variance;
    private double[] gradGamma;
    private double[] gradBeta;
    private double epsilon = 1e-5;
    private double momentum = 0.9;
    private double[] runningMean;
    private double[] runningVariance;
    private boolean isTraining = true;

    public BatchNormalizationLayer(int size) {
        gamma = new double[size];
        beta = new double[size];
        gradGamma = new double[size];
        gradBeta = new double[size];
        runningMean = new double[size];
        runningVariance = new double[size];
        Arrays.fill(gamma, 1.0);
        Arrays.fill(beta, 0.0);
    }

    @Override
    public double[] forward(double[] input) {
        this.input = input;

        if (isTraining) {
            // Compute mean
            mean = new double[input.length];
            double sum = 0;
            for (double v : input) sum += v;
            double batchMean = sum / input.length;
            Arrays.fill(mean, batchMean);

            // Compute variance
            variance = new double[input.length];
            double sqSum = 0;
            for (double v : input) sqSum += (v - batchMean) * (v - batchMean);
            double batchVariance = sqSum / input.length;
            Arrays.fill(variance, batchVariance);

            // Normalize
            normalizedInput = new double[input.length];
            for (int i = 0; i < input.length; i++) {
                normalizedInput[i] = (input[i] - mean[i]) / Math.sqrt(variance[i] + epsilon);
            }

            // Update running mean and variance
            for (int i = 0; i < input.length; i++) {
                runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[i];
                runningVariance[i] = momentum * runningVariance[i] + (1 - momentum) * variance[i];
            }
        } else {
            // Use running mean and variance
            normalizedInput = new double[input.length];
            for (int i = 0; i < input.length; i++) {
                normalizedInput[i] = (input[i] - runningMean[i]) / Math.sqrt(runningVariance[i] + epsilon);
            }
        }

        // Scale and shift
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = gamma[i] * normalizedInput[i] + beta[i];
        }
        return output;
    }

    @Override
    public double[] backward(double[] gradOutput) {
        int N = input.length;

        // Compute gradients
        for (int i = 0; i < N; i++) {
            gradGamma[i] = gradOutput[i] * normalizedInput[i];
            gradBeta[i] = gradOutput[i];
        }

        // Backprop through normalization
        double[] gradInput = new double[N];
        for (int i = 0; i < N; i++) {
            gradInput[i] = (1.0 / N) * gamma[i] / Math.sqrt(variance[i] + epsilon) *
                    (N * gradOutput[i] - gradBeta[i] - normalizedInput[i] * gradGamma[i]);
        }
        return gradInput;
    }

    @Override
    public void updateWeights(double learningRate) {
        for (int i = 0; i < gamma.length; i++) {
            gamma[i] -= learningRate * gradGamma[i];
            beta[i] -= learningRate * gradBeta[i];
        }
    }
}