import java.util.function.Function;

// This class provides various neural network activation functions and their derivatives.
public class ActivationFunctions {
    public static Function<Double, Double> tanh = Math::tanh;
    public static Function<Double, Double> tanhDerivative = x -> 1 - Math.pow(Math.tanh(x), 2);

    public static Function<Double, Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
    public static Function<Double, Double> sigmoidDerivative = x -> {
        double s = sigmoid.apply(x);
        return s * (1 - s);
    };

    public static Function<Double, Double> relu = x -> Math.max(0, x);
    public static Function<Double, Double> reluDerivative = x -> x > 0 ? 1.0 : 0.0;

    public static Function<Double, Double> leakyReLU(double alpha) {
        return x -> x > 0 ? x : alpha * x;
    }

    public static Function<Double, Double> leakyReLUDerivative(double alpha) {
        return x -> x > 0 ? 1.0 : alpha;
    }

    public static Function<Double, Double> elu(double alpha) {
        return x -> x >= 0 ? x : alpha * (Math.exp(x) - 1);
    }

    public static Function<Double, Double> eluDerivative(double alpha) {
        return x -> x >= 0 ? 1 : alpha * Math.exp(x);
    }

    public static Function<Double, Double> softplus = x -> Math.log(1 + Math.exp(x));
    public static Function<Double, Double> softplusDerivative = x -> 1 / (1 + Math.exp(-x));

    public static Function<Double, Double> softsign = x -> x / (1 + Math.abs(x));
    public static Function<Double, Double> softsignDerivative = x -> 1 / Math.pow(1 + Math.abs(x), 2);
}
