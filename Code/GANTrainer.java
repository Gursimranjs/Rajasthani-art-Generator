import java.util.Random;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;

// This class trains a Generative Adversarial Network (GAN) using a generator and a discriminator.
// It manages the training process, including data loading, noise generation, loss calculation,
// backpropagation, and weight updates. The trainer applies techniques like label smoothing,
// noise augmentation, and learning rate decay for stable training and logs metrics to a CSV file.

public class GANTrainer {
    private Generator generator;
    private Discriminator discriminator;
    private DataLoader dataLoader;
    private int noiseSize;
    private double generatorLearningRate;
    private double discriminatorLearningRate;
    private double generatorDecay;
    private double discriminatorDecay;
    private int batchSize;

    public GANTrainer(Generator generator, Discriminator discriminator, DataLoader dataLoader, int noiseSize, double generatorLearningRate, double discriminatorLearningRate, int batchSize) {
        this.generator = generator;
        this.discriminator = discriminator;
        this.dataLoader = dataLoader;
        this.noiseSize = noiseSize;
        this.generatorLearningRate = generatorLearningRate;
        this.discriminatorLearningRate = discriminatorLearningRate;
        this.generatorDecay = 0.00001;
        this.discriminatorDecay = 0.00001;
        this.batchSize = batchSize;
    }

    public void train(int epochs) {
        double[][] images = dataLoader.getImages();
        if (images.length == 0) {
            System.err.println("No images available for training.");
            return;
        }

        int numBatches = images.length / batchSize;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            shuffleData(images);

            double totalDiscriminatorLoss = 0;
            double totalGeneratorLoss = 0;

            for (int batch = 0; batch < numBatches; batch++) {
                // Prepare batches
                double[][] realBatch = Arrays.copyOfRange(images, batch * batchSize, (batch + 1) * batchSize);
                double[][] noiseBatch = generateNoiseBatch(batchSize, noiseSize);
                double[][] fakeBatch = generateFakeBatch(noiseBatch);

                // Add noise to real images
                for (int i = 0; i < realBatch.length; i++) {
                    realBatch[i] = addNoise(realBatch[i], 0.05); // Small noise for stability
                }

                // Train Discriminator more frequently to maintain balance
                if (batch % 1 == 0) {
                    totalDiscriminatorLoss += trainDiscriminator(realBatch, fakeBatch);
                }

                // Train Generator
                totalGeneratorLoss += trainGenerator(noiseBatch);
            }

            // Print losses and update learning rates
            System.out.printf("Epoch %d/%d, Discriminator Loss: %.6f, Generator Loss: %.6f%n",
                    epoch, epochs, totalDiscriminatorLoss / numBatches, totalGeneratorLoss / numBatches);
            logMetrics(epoch, totalDiscriminatorLoss / numBatches, totalGeneratorLoss / numBatches);

            generatorLearningRate -= generatorDecay;
            discriminatorLearningRate -= discriminatorDecay;

            // Save generated image
            saveGeneratedImage(epoch);
        }
        System.out.println("GAN training completed.");
    }

    private double trainDiscriminator(double[][] realBatch, double[][] fakeBatch) {
        double totalLoss = 0;
        double[] realLabels = new double[batchSize];
        Arrays.fill(realLabels, 0.9); // Label smoothing for real images
        double[] fakeLabels = new double[batchSize];
        Arrays.fill(fakeLabels, 0.1); // Less confident fake labels

        // Randomly flip real and fake labels for discriminator training
        Random rand = new Random();
        if (rand.nextDouble() < 0.1) {
            double[] temp = realLabels;
            realLabels = fakeLabels;
            fakeLabels = temp;
        }

        // Train on real images
        for (int i = 0; i < realBatch.length; i++) {
            totalLoss += discriminator.train(realBatch[i], realLabels[i]);
        }

        // Train on fake images
        for (int i = 0; i < fakeBatch.length; i++) {
            totalLoss += discriminator.train(fakeBatch[i], fakeLabels[i]);
        }

        discriminator.updateWeights(discriminatorLearningRate);
        return totalLoss / (2 * batchSize);
    }

    private double trainGenerator(double[][] noiseBatch) {
        double totalLoss = 0;

        for (int i = 0; i < noiseBatch.length; i++) {
            double[] fakeImage = generator.forward(noiseBatch[i]);
            double[] discriminatorOutput = discriminator.forward(fakeImage);

            // Target for generator is to make discriminator output close to 1
            double[] gradOutput = new double[1];
            gradOutput[0] = 2 * (discriminatorOutput[0] - 1); // Gradient of L2 loss

            // Backpropagation through discriminator and generator
            discriminator.backward(gradOutput);
            generator.backward(discriminator.getInputGradients());

            // Compute loss
            totalLoss += Math.pow(discriminatorOutput[0] - 1, 2);
        }

        generator.updateWeights(generatorLearningRate);
        return totalLoss / noiseBatch.length;
    }

    private void saveGeneratedImage(int epoch) {
        double[] noise = generateNoise();
        double[] generatedImage = generator.forward(noise);
        String filePath = "Output/generated_epoch_" + epoch + ".png";
        ImageGenerator.saveImage(generatedImage, dataLoader.getImageWidth(), dataLoader.getImageHeight(), filePath);
        System.out.printf("Generated image saved for epoch %d: %s%n", epoch, filePath);
    }

    private double[][] generateNoiseBatch(int batchSize, int noiseSize) {
        double[][] noiseBatch = new double[batchSize][noiseSize];
        Random rand = new Random();
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < noiseSize; j++) {
                noiseBatch[i][j] = rand.nextGaussian();
            }
        }
        return noiseBatch;
    }

    private double[][] generateFakeBatch(double[][] noiseBatch) {
        double[][] fakeBatch = new double[noiseBatch.length][];
        for (int i = 0; i < noiseBatch.length; i++) {
            fakeBatch[i] = generator.forward(noiseBatch[i]); // Generate fake images
        }
        return fakeBatch;
    }

    double[] generateNoise() {
        double[] noise = new double[noiseSize];
        Random rand = new Random();
        for (int i = 0; i < noiseSize; i++) {
            noise[i] = rand.nextGaussian();
        }
        return noise;
    }

    private void shuffleData(double[][] data) {
        Random rand = new Random();
        for (int i = data.length - 1; i > 0; i--) {
            int index = rand.nextInt(i + 1);
            double[] temp = data[index];
            data[index] = data[i];
            data[i] = temp;
        }
    }

    private double[] addNoise(double[] image, double noiseLevel) {
        double[] noisyImage = new double[image.length];
        Random rand = new Random();
        for (int i = 0; i < image.length; i++) {
            noisyImage[i] = image[i] + noiseLevel * rand.nextGaussian();
        }
        return noisyImage;
    }

    private void logMetrics(int epoch, double dLoss, double gLoss) {
        try (FileWriter writer = new FileWriter("training_log.csv", true)) {
            writer.append(epoch + "," + dLoss + "," + gLoss + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
