public class Main {
    public static void main(String[] args) {
        try {
            int imageWidth = 64;
            int imageHeight = 64;
            int noiseSize = 100;
            // Modify learning rates
            double generatorLearningRate = 0.0001; // Reduced learning rate for generator
            double discriminatorLearningRate = 0.0002; // Reduced learning rate for discriminator


            int batchSize = 32;
            int epochs = 100;

            System.out.println("Initializing DataLoader...");
            DataLoader dataLoader = new DataLoader("Code/Data/Images", imageWidth, imageHeight);
            System.out.println("Images loaded: " + dataLoader.getImages().length);

            if (dataLoader.getImages().length == 0) {
                System.err.println("No images loaded. Exiting.");
                return;
            }

            System.out.println("Initializing Generator and Discriminator...");
            Generator generator = new Generator(noiseSize, imageWidth * imageHeight * 3);
            Discriminator discriminator = new Discriminator(imageWidth * imageHeight * 3);

            System.out.println("Starting GAN training...");
            GANTrainer trainer = new GANTrainer(generator, discriminator, dataLoader, noiseSize, generatorLearningRate, discriminatorLearningRate, batchSize);
            trainer.train(epochs);

            System.out.println("Training completed. Generating final image...");
            double[] noise = trainer.generateNoise();
            double[] generatedImage = generator.forward(noise);
            String finalImagePath = "Output/generated_image_final.png";
            ImageGenerator.saveImage(generatedImage, imageWidth, imageHeight, finalImagePath);
            System.out.println("Final generated image saved as " + finalImagePath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}