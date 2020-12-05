
import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy import ndimage
from skimage import img_as_ubyte
from pathlib import Path



if __name__ == '__main__':
    img = img_as_ubyte(img)
    # img = cv2.imread('deneme.png')
    showPlot(img)
    plt.set_cmap(plt.get_cmap(name='gray'))
    if len(img.shape) == 3:
        img = rgb_to_gray(img)
        saveOutput(img, "gray_" + name)
