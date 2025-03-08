from PIL import Image
import numpy as np

def main():
    image = Image.open('./foty/image.png')
    pixels = np.array(image)

    R = pixels[:, :, 0]

if __name__ == "__main__":
    main()