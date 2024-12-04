import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class DataLoader {
    private double[][] images;
    private int imageWidth;
    private int imageHeight;

    public DataLoader(String folderPath, int imageWidth, int imageHeight) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        System.out.println("Loading images from: " + folderPath);
        loadImages(folderPath);
    }

    private void loadImages(String folderPath) {
        File folder = new File(folderPath);
        File[] listOfFiles = folder.listFiles();
        if (listOfFiles == null || listOfFiles.length == 0) {
            System.err.println("No images found in the directory: " + folderPath);
            images = new double[0][];
            return;
        }

        images = new double[listOfFiles.length * 4][imageWidth * imageHeight * 3]; // Augmented data

        int index = 0;
        int loadedImages = 0;
        for (File file : listOfFiles) {
            try {
                BufferedImage img = ImageIO.read(file);
                if (img == null) {
                    System.err.println("Failed to load image: " + file.getName());
                    continue;
                }
                BufferedImage resized = resizeImage(img, imageWidth, imageHeight);
                images[index++] = imageToArray(resized);
                images[index++] = imageToArray(flipImage(resized)); // Augmented image (flipped)
                images[index++] = imageToArray(rotateImage(resized, 45)); // Augmented image (rotated 45 degrees)
                images[index++] = imageToArray(adjustBrightness(resized, 1.2)); // Augmented image (brightness adjusted)
                loadedImages += 4;
            } catch (IOException e) {
                System.err.println("Error reading image file: " + file.getName());
                e.printStackTrace();
            }
        }

        if (loadedImages == 0) {
            System.err.println("No images were successfully loaded.");
        } else {
            System.out.println("Total images loaded and augmented: " + loadedImages);
        }
    }

    private BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        g.dispose();
        return resized;
    }

    private BufferedImage flipImage(BufferedImage img) {
        BufferedImage flipped = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < img.getHeight(); y++) {
            for (int x = 0; x < img.getWidth(); x++) {
                flipped.setRGB(img.getWidth() - x - 1, y, img.getRGB(x, y));
            }
        }
        return flipped;
    }

    private BufferedImage rotateImage(BufferedImage img, double angle) {
        int width = img.getWidth();
        int height = img.getHeight();
        BufferedImage rotated = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = rotated.createGraphics();
        g.rotate(Math.toRadians(angle), width / 2.0, height / 2.0);
        g.drawImage(img, 0, 0, null);
        g.dispose();
        return rotated;
    }

    private BufferedImage adjustBrightness(BufferedImage img, double brightnessFactor) {
        BufferedImage brightened = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < img.getHeight(); y++) {
            for (int x = 0; x < img.getWidth(); x++) {
                Color originalColor = new Color(img.getRGB(x, y));
                int r = Math.min((int) (originalColor.getRed() * brightnessFactor), 255);
                int g = Math.min((int) (originalColor.getGreen() * brightnessFactor), 255);
                int b = Math.min((int) (originalColor.getBlue() * brightnessFactor), 255);
                Color newColor = new Color(r, g, b);
                brightened.setRGB(x, y, newColor.getRGB());
            }
        }
        return brightened;
    }

    private double[] imageToArray(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[] result = new double[width * height * 3];
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = img.getRGB(x, y);
                result[index++] = ((rgb >> 16) & 0xFF) / 127.5 - 1; // Red
                result[index++] = ((rgb >> 8) & 0xFF) / 127.5 - 1;  // Green
                result[index++] = (rgb & 0xFF) / 127.5 - 1;         // Blue
            }
        }
        return result;
    }

    public double[][] getImages() {
        return images;
    }

    public int getImageWidth() {
        return imageWidth;
    }

    public int getImageHeight() {
        return imageHeight;
    }
}
