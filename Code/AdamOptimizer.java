public class AdamOptimizer implements Optimizer {
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double learningRate;
    private double epsilon = 1e-8;
    private double[][] mWeights, vWeights;
    private double[] mBiases, vBiases;
    private int t = 0;

    public AdamOptimizer(double learningRate, int outputSize, int inputSize) {
        this.learningRate = learningRate;
        mWeights = new double[outputSize][inputSize];
        vWeights = new double[outputSize][inputSize];
        mBiases = new double[outputSize];
        vBiases = new double[outputSize];
    }

    @Override
    public void update(double[][] weights, double[][] gradWeights) {
        t++;
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                // Update biased first moment estimate
                mWeights[i][j] = beta1 * mWeights[i][j] + (1 - beta1) * gradWeights[i][j];
                // Update biased second raw moment estimate
                vWeights[i][j] = beta2 * vWeights[i][j] + (1 - beta2) * gradWeights[i][j] * gradWeights[i][j];
                // Compute bias-corrected first moment estimate
                double mHat = mWeights[i][j] / (1 - Math.pow(beta1, t));
                // Compute bias-corrected second raw moment estimate
                double vHat = vWeights[i][j] / (1 - Math.pow(beta2, t));
                // Update weights
                weights[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }

    @Override
    public void update(double[] biases, double[] gradBiases) {
        t++;
        for (int i = 0; i < biases.length; i++) {
            mBiases[i] = beta1 * mBiases[i] + (1 - beta1) * gradBiases[i];
            vBiases[i] = beta2 * vBiases[i] + (1 - beta2) * gradBiases[i] * gradBiases[i];
            double mHat = mBiases[i] / (1 - Math.pow(beta1, t));
            double vHat = vBiases[i] / (1 - Math.pow(beta2, t));
            biases[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }
}