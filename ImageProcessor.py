from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


    def grayscale(self, method="luminosity", processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
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

        # self.show(pixels)
        self.processed_pixels = pixels


    def adjust_brightness(self, brightnessFactor=0, processed=False):
        # self.processed_pixels = np.clip(self.pixels.astype(np.int16) + brightnessFactor, 0, 255).astype(np.uint8)
        # self.show(self.processed_pixels)

        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        adjusted_R = np.clip(R.astype(np.int16) + brightnessFactor, 0, 255).astype(np.uint8)
        adjusted_G = np.clip(G.astype(np.int16) + brightnessFactor, 0, 255).astype(np.uint8)
        adjusted_B = np.clip(B.astype(np.int16) + brightnessFactor, 0, 255).astype(np.uint8)

        self.processed_pixels = np.stack([adjusted_R, adjusted_G, adjusted_B, A], axis=-1) if A is not None else np.stack([adjusted_R, adjusted_G, adjusted_B], axis=-1)

    def contrast_correction(self, contrastFactor: int = 0, processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()
        
        F = (259 * (contrastFactor + 255)) / (255 * (259 - contrastFactor))
        
        adjust = lambda channel: np.clip(F * (channel.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
        
        adjusted_R = adjust(R)
        adjusted_G = adjust(G)
        adjusted_B = adjust(B)
        
        self.processed_pixels = np.stack([adjusted_R, adjusted_G, adjusted_B, A], axis=-1) if A is not None else np.stack([adjusted_R, adjusted_G, adjusted_B], axis=-1)


    def negative(self, processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()
        
        self.processed_pixels = np.stack([255 - R, 255 - G, 255 - B, A], axis=-1) if A is not None else np.stack([255 - R, 255 - G, 255 - B], axis=-1)


    def binarize(self, threshold, method="luminosity", processed=True):
        self.grayscale(method)
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        R_new = np.where(R > threshold, 255, 0).astype(np.uint8)
        G_new = np.where(G > threshold, 255, 0).astype(np.uint8)
        B_new = np.where(B > threshold, 255, 0).astype(np.uint8)

        if A is not None:
            self.processed_pixels = np.stack([R_new, G_new, B_new, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R_new, G_new, B_new], axis=-1)


    def filter(self, method="average", processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        methods = {"average": np.ones((3, 3)) * 1/9,
                "gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
                "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])}

        mask = methods[method]
        d = mask.shape[0] // 2

        n = R.shape[0]
        m = R.shape[1]
        R_new = np.zeros((n, m), dtype=np.float32)
        G_new = np.zeros((n, m), dtype=np.float32)
        B_new = np.zeros((n, m), dtype=np.float32)

        for i in range(d, n-d):
            for j in range(d, m-d):
                sub_R = R[i-d:i+d+1, j-d:j+d+1]
                sub_G = G[i-d:i+d+1, j-d:j+d+1] 
                sub_B = B[i-d:i+d+1, j-d:j+d+1] 

                R_new[i, j] = np.sum(np.multiply(sub_R, mask))
                G_new[i, j] = np.sum(np.multiply(sub_G, mask))
                B_new[i, j] = np.sum(np.multiply(sub_B, mask))

        R_new = np.clip(R_new, 0, 255).astype(np.uint8)
        G_new = np.clip(G_new, 0, 255).astype(np.uint8)
        B_new = np.clip(B_new, 0, 255).astype(np.uint8)

        if A is not None:
            self.processed_pixels = np.stack([R_new, G_new, B_new, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R_new, G_new, B_new], axis=-1)


    def filter2(self, method="average", processed=False):
        """much faster than filter1"""

        from scipy import signal 
        
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        methods = {
            "average": np.ones((3, 3)) * 1/9,
            "gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        }

        mask = methods[method]
        
        R_new = signal.convolve2d(R, mask, mode='same', boundary='symm')
        G_new = signal.convolve2d(G, mask, mode='same', boundary='symm')
        B_new = signal.convolve2d(B, mask, mode='same', boundary='symm')

        R_new = np.clip(R_new, 0, 255).astype(np.uint8)
        G_new = np.clip(G_new, 0, 255).astype(np.uint8)
        B_new = np.clip(B_new, 0, 255).astype(np.uint8)

        if A is not None:
            self.processed_pixels = np.stack([R_new, G_new, B_new, A], axis=-1)
        else:
            self.processed_pixels = np.stack([R_new, G_new, B_new], axis=-1)

    def make_histogram(self, processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()

        R_flat = R.ravel()
        G_flat = G.ravel()
        B_flat = B.ravel()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), sharey=True)

        ax1.hist(R_flat, bins=256, range=(0, 255), color='r', alpha=0.8, edgecolor='none')
        ax1.set_title('Red Channel', color='r')
        ax1.set_xlim(0, 256)

        ax2.hist(G_flat, bins=256, range=(0, 255), color='g', alpha=0.8, edgecolor='none')
        ax2.set_title('Green Channel', color='g')
        ax2.set_xlim(0, 256)

        ax3.hist(B_flat, bins=256, range=(0, 255), color='b', alpha=0.8, edgecolor='none')
        ax3.set_title('Blue Channel', color='b')
        ax3.set_xlim(0, 256)

        fig.suptitle('RGB Histograms by Channel', fontsize=16)
        
        plt.tight_layout()

        return fig
        
    def horizontal_projection(self, processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()
        
        # Conversion to grayscale for projection calculation
        if np.array_equal(R, G) and np.array_equal(G, B):
            gray = R
        else:
            gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

        # binarize
        threshold = 127
        gray = np.where(gray > threshold, 255, 0).astype(np.uint8)
        
        horizontal_proj = np.sum(gray, axis=1) / gray.shape[1]
        
        fig, ax = plt.subplots(figsize=(6, 8))
        
        ax.barh(np.arange(len(horizontal_proj)), horizontal_proj, height=1, color='black', alpha=0.7)
        ax.set_title('Horizontal Projection')
        ax.set_xlim(0, 255)
        ax.set_xlabel('Average intensity')
        ax.invert_yaxis() 
        
        plt.tight_layout()
        
        return horizontal_proj, fig

    def vertical_projection(self, processed=False):
        if processed:
            R, G, B, A = self.get_RGBA(processed=True)
        else:
            R, G, B, A = self.get_RGBA()
        
        # Conversion to grayscale for projection calculation
        if np.array_equal(R, G) and np.array_equal(G, B):
            gray = R
        else:
            gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

        # binarize
        threshold = 127
        gray = np.where(gray > threshold, 255, 0).astype(np.uint8)
        
        vertical_proj = np.sum(gray, axis=0) / gray.shape[0] 
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.bar(np.arange(len(vertical_proj)), vertical_proj, width=1, color='black', alpha=0.7)
        ax.set_title('Vertical Projection')
        ax.set_ylim(0, 255)
        ax.set_ylabel('Average intensity')
        
        plt.tight_layout()
        
        return vertical_proj, fig

    def show(self, pixels=None):
        if pixels is None:
            pixels = self.pixels
        img = Image.fromarray(pixels)
        img.show()

    def show_processed(self):
        img = Image.fromarray(self.processed_pixels)
        img.show()

    def save(self, output_path):
        img = Image.fromarray(self.processed_pixels)
        img.save(output_path)

    def reset(self):
        self.processed_pixels = self.pixels.copy()        

if __name__ == "__main__":
    processor = ImageProcessor('foty/chillguy.jpeg')
    # lenka = ImageProcessor('./foty/lenka.png')
    # processor.grayscale(processed=True)
    # processor.adjust_brightness(50, processed=True)
    # processor.contrast_correction(50, processed=True)
    # processor.show_processed()
    # processor.reset()
    # processor.show_processed()
    processor.horizontal_projection()
    
    # lenka.binarize(120)