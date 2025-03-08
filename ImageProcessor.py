from PIL import Image
import numpy as np

class ImageProcessor:

    def __init__(self, image_path):
        self.image = Image.open(image_path).convert("RGBA")
        self.pixels = np.array(self.image)
        self.processed_pixels = self.pixels.copy()


    def get_RGBA(self, processed=False):
        if processed:
            if self.processed_pixels.shape[-1] == 4:
                return self.processed_pixels[:, :, 0], self.processed_pixels[:, :, 1], self.processed_pixels[:, :, 2], self.processed_pixels[:, :, 3]
            else:
                return self.processed_pixels[:, :, 0], self.processed_pixels[:, :, 1], self.processed_pixels[:, :, 2]
                
        if self.pixels.shape[-1] == 4:
            return self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2], self.pixels[:, :, 3]
        return self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2], None 


    def grayscale(self, method="luminosity"):
        R, G, B, A = self.get_RGBA()
         
        methods = {
            "average": ((R.astype(np.uint16) + G.astype(np.uint16) + B.astype(np.uint16)) // 3).astype(np.uint8),
            "luminosity": (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8),
            "luminance": (0.2126 * R + 0.7152 * G + 0.0722 * B).astype(np.uint8),
            "desaturation": ((np.maximum(R, G, B).astype(np.uint16) + np.minimum(R, G, B).astype(np.uint16)) // 2).astype(np.uint8),
            "decomposition-max": np.maximum(R, G, B).astype(np.uint8),
            "decomposition-min": np.minimum(R, G, B).astype(np.uint8),
        }

        if method not in methods:
            raise ValueError("Unknown grayscale method!")

        gray = methods[method]
        pixels = np.stack([gray, gray, gray, A], axis=-1) if A is not None else np.stack([gray] * 3, axis=-1)

        self.show(pixels)
        self.processed_pixels = pixels


    def adjust_brightness(self, brightnessFactor=0):
        self.processed_pixels = np.clip(self.pixels.astype(np.int16) + brightnessFactor, 0, 255).astype(np.uint8)
        self.show(self.processed_pixels)


    def contrast_correction(self, contrastFactor: int = 0):
        R, G, B, A = self.get_RGBA()
        
        F = (259 * (contrastFactor + 255)) / (255 * (259 - contrastFactor))
        
        adjust = lambda channel: np.clip(F * (channel.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
        
        adjusted_R = adjust(R)
        adjusted_G = adjust(G)
        adjusted_B = adjust(B)
        
        self.processed_pixels = np.stack([adjusted_R, adjusted_G, adjusted_B, A], axis=-1) if A is not None else np.stack([adjusted_R, adjusted_G, adjusted_B], axis=-1)
        
        self.show(self.processed_pixels) 


    def negative(self):
        R, G, B, A = self.get_RGBA()
        
        self.processed_pixels = np.stack([255 - R, 255 - G, 255 - B, A], axis=-1) if A is not None else np.stack([255 - R, 255 - G, 255 - B], axis=-1)

        self.show(self.processed_pixels) 


    def binarize(self, threshold, method="luminosity"):
        self.grayscale(method)
        R, G, B, A = self.get_RGBA(processed=True)

        R_new = np.where(R > threshold, 255, 0).astype(np.uint8)
        G_new = np.where(G > threshold, 255, 0).astype(np.uint8)
        B_new = np.where(B > threshold, 255, 0).astype(np.uint8)

        if A is not None:
            self.processed_pixels = np.stack([R_new, G_new, B_new, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R_new, G_new, B_new], axis=-1)

        self.show(self.processed_pixels)


    def show(self, pixels=None):
        if pixels is None:
            pixels = self.pixels
        img = Image.fromarray(pixels)
        img.show()

    def save(self, output_path):
        img = Image.fromarray(self.processed_pixels)
        img.save(output_path)


if __name__ == "__main__":
    processor = ImageProcessor('foty/chillguy.jpeg')
    lenka = ImageProcessor('./foty/image.png')
    # processor.grayscale(method="average")
    # processor.grayscale(method="luminosity")
    # processor.grayscale(method="luminance")
    # processor.grayscale(method="desaturation")
    # processor.grayscale(method="decomposition-max")
    # processor.grayscale(method="decomposition-min")
    # processor.adjust_brightness(brightnessFactor=60)
    # processor.contrast_correction(128)
    lenka.binarize(120)