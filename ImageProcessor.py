from PIL import Image
import numpy as np

class ImageProcessor:

    def __init__(self, image_path):
        self.image = Image.open(image_path).convert("RGBA")
        self.pixels = np.array(self.image)
        self.processed_pixels = self.pixels.copy()

    def grayscale(self, method="luminosity"):
        channels = self.pixels.shape[-1]
        has_alpha = channels == 4

        if has_alpha:
            R, G, B, A = self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2], self.pixels[:, :, 3]
        else:
            R, G, B = self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2]
            A = None  

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
        pixels = np.stack([gray, gray, gray, A], axis=-1) if has_alpha else np.stack([gray] * 3, axis=-1)

        self.show(pixels)
        self.processed_pixels = pixels

    def adjust_brightness(self, brightnessFactor=0):
        # ten chyba nic nie robi z ciemnymi bo mnozenie lepiej dziala na jasne obszary niz na ciemne
        # self.processed_pixels = np.clip(self.pixels.astype(np.float32) * (1 + brightnessFactor), 0, 255).astype(np.uint8)
        
        # opcja z poprostu dodaniem wartosci
        self.processed_pixels = np.clip(self.pixels.astype(np.int16) + brightnessFactor, 0, 255).astype(np.uint8)
        
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
    # processor.grayscale(method="average")
    # processor.grayscale(method="luminosity")
    # processor.grayscale(method="luminance")
    # processor.grayscale(method="desaturation")
    # processor.grayscale(method="decomposition-max")
    # processor.grayscale(method="decomposition-min")
    # processor.adjust_brightness(brightnessFactor=1.5)
    processor.adjust_brightness(100)

        