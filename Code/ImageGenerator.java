import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class ImageGenerator {
    public static void saveImage(double[] imageArray, int width, int height, String filePath) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (int) ((imageArray[index++] + 1) * 127.5); // Scale back to [0, 255]
                int g = (int) ((imageArray[index++] + 1) * 127.5);
                int b = (int) ((imageArray[index++] + 1) * 127.5);

                // Clamp values to [0, 255]
                r = Math.max(0, Math.min(255, r));
                g = Math.max(0, Math.min(255, g));
                b = Math.max(0, Math.min(255, b));

                int rgb = (r << 16) | (g << 8) | b;
                image.setRGB(x, y, rgb);
            }
        }
        try {
            File outputFile = new File(filePath);
            // Ensure parent directories exist
            outputFile.getParentFile().mkdirs();
            ImageIO.write(image, "png", outputFile);
            //System.out.println("Image saved to: " + filePath);
        } catch (Exception e) {
            System.err.println("Failed to save image to: " + filePath);
            e.printStackTrace();
        }
    }
}