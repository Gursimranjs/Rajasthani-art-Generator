import java.util.function.Function;

public class ActivationFunctions {
    // Tanh activation function and its derivative
    public static Function<Double, Double> tanh = Math::tanh;
    public static Function<Double, Double> tanhDerivative = x -> 1 - Math.pow(Math.tanh(x), 2);

    // Sigmoid activation function and its derivative (with optimized re-use)
    public static Function<Double, Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
    public static Function<Double, Double> sigmoidDerivative = x -> {
        double s = sigmoid.apply(x); // Reuse sigmoid value to optimize
        return s * (1 - s);
    };

    // ReLU activation function and its derivative
    public static Function<Double, Double> relu = x -> Math.max(0, x);
    public static Function<Double, Double> reluDerivative = x -> x > 0 ? 1.0 : 0.0;

    // Parametrizable Leaky ReLU activation function and its derivative
    public static Function<Double, Double> leakyReLU(double alpha) {
        return x -> x > 0 ? x : alpha * x;
    }

    public static Function<Double, Double> leakyReLUDerivative(double alpha) {
        return x -> x > 0 ? 1.0 : alpha;
    }

    // ELU (Exponential Linear Unit) activation function and its derivative
    public static Function<Double, Double> elu(double alpha) {
        return x -> x >= 0 ? x : alpha * (Math.exp(x) - 1);
    }

    public static Function<Double, Double> eluDerivative(double alpha) {
        return x -> x >= 0 ? 1 : alpha * Math.exp(x);
    }

    // Softplus activation function and its derivative
    public static Function<Double, Double> softplus = x -> Math.log(1 + Math.exp(x));
    public static Function<Double, Double> softplusDerivative = x -> 1 / (1 + Math.exp(-x));

    // Softsign activation function and its derivative
    public static Function<Double, Double> softsign = x -> x / (1 + Math.abs(x));
    public static Function<Double, Double> softsignDerivative = x -> 1 / Math.pow(1 + Math.abs(x), 2);
}
