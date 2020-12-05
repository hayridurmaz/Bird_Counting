
import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy import ndimage
from skimage import img_as_ubyte
from pathlib import Path


def rgb_to_gray(img):
    np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    # rates for grayscale image
    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    grayImage = np.array(img)
    grayImage.setflags(write=1)
    grayImage[:, :, 0] = Avg
    return grayImage[:, :, 0]


def showPlot(img):
    plt.imshow(img)
    plt.show()

def saveOutput(img, name):
    Path("output").mkdir(parents=True, exist_ok=True)
    mpimg.imsave("output/" + name, img)

def readImage(path):
    img = mpimg.imread("images/" + path)
    return img

if __name__ == '__main__':
    name="bird_1.jpg"
    img= readImage(name)
    img = img_as_ubyte(img)
    showPlot(img)
    plt.set_cmap(plt.get_cmap(name='gray'))
    if len(img.shape) == 3:
        img = rgb_to_gray(img)
        saveOutput(img, "gray_" + name)
