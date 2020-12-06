import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy import ndimage
from skimage import img_as_ubyte
from pathlib import Path

from Watershed import Watershed


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


def showPlot(img, cmap="gray"):
    plt.set_cmap(cmap)
    plt.imshow(img)
    plt.show()


def saveOutput(img, name):
    Path("output").mkdir(parents=True, exist_ok=True)
    mpimg.imsave("output/" + name, img)


def readImage(path):
    img = mpimg.imread("images/" + path)
    return img


def globalThresholding(image, threshold=127):
    img = np.copy(image)
    img[img > threshold] = 0
    img[img != 0] = 255
    return img


def thresholdIntegral(inputMat, s, T=0.15):
    # outputMat=np.uint8(np.ones(inputMat.shape)*255)
    outputMat = np.zeros(inputMat.shape)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)

    s2 = int(S / 4)

    for i in range(nRows):
        y1 = i - s2
        y2 = i + s2

        if (y1 < 0):
            y1 = 0
        if (y2 >= nRows):
            y2 = nRows - 1

        for j in range(nCols):
            x1 = j - s2
            x2 = j + s2

            if (x1 < 0):
                x1 = 0
            if (x2 >= nCols):
                x2 = nCols - 1
            count = (x2 - x1) * (y2 - y1)

            sum = s[y2][x2] - s[y2][x1] - s[y1][x2] + s[y1][x1]

            if ((int)(inputMat[i][j] * count) < (int)(sum * (1.0 - T))):
                outputMat[i][j] = 255
                # print(i,j)
            # else:
            #     outputMat[j][i] = 0
    return outputMat


def getOtsuThreshold(im):
    size = 256
    buckets = np.zeros([size])
    imy = im.shape[0]
    imx = im.shape[1]
    image_size = imx * imy
    for i in range(imy):
        for j in range(imx):
            buckets[im[i][j]] += 1

    in_class_variance_list = []

    for i in range(1, size):
        wf = np.sum(buckets[i:]) / image_size
        wb = 1 - wf
        mf = np.sum(np.dot(range(i, size), buckets[i:])) / np.sum(buckets[i:])
        mb = np.sum(np.dot(range(i), buckets[:i])) / np.sum(buckets[:i])
        varf = np.sum(np.dot(np.power(range(i, size) - mf, 2), buckets[i:size])) / np.sum(buckets[i:])
        varb = np.sum(np.dot(np.power(range(i) - mb, 2), buckets[:i])) / np.sum(buckets[:i])
        in_class_variance = varf * wf + varb * wb
        in_class_variance_list.append(in_class_variance)
    return in_class_variance_list.index(min(in_class_variance_list))


if __name__ == '__main__':
    birdnames = ["bird_1.jpg", "bird_2.jpg", "bird_3.bmp"]
    # birdnames = ["bird_1.jpg"]
    plt.set_cmap(cmap="gray")

    for name in birdnames:
        img = readImage(name)
        img = img_as_ubyte(img)
        if len(img.shape) == 3:
            img = rgb_to_gray(img)
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()

        # ret2, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_threshold = getOtsuThreshold(img)
        print("otsu threshold for " + name + " is " + str(otsu_threshold))
        thresholded = globalThresholding(img, otsu_threshold)
        showPlot(thresholded)



        
        # kernel = np.array([[1, 0, 0, 0, 0],
        #                    [0, 1, 0, 0, 0],
        #                    [0, 0, 1, 0, 0],
        #                    [0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 1]], dtype=np.uint8)
        #
        # erosion = cv2.erode(thresholded, kernel, iterations=1)
        # showPlot(erosion)
        # dilotion = cv2.dilate(thresholded, kernel, iterations=1)
        # showPlot(dilotion)
        #
        # kernel = np.ones((5, 5), np.uint8)
        # dilotion = cv2.dilate(thresholded, kernel, iterations=1)
        # showPlot(dilotion)

        # opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        # showPlot(opening)

        # closing = cv2.morphologyEx(thresholded, cv2.MORPH_GRADIENT, kernel)
        # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
        # showPlot(closing)
