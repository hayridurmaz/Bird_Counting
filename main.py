import copy
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

weak = np.int32(75)
strong = np.int32(255)
lowThresholdRatio = 0.05
highThresholdRatio = 20

sys.setrecursionlimit(10000)


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


# def thresholdIntegral(inputMat, s, T=0.15):
#     # outputMat=np.uint8(np.ones(inputMat.shape)*255)
#     outputMat = np.zeros(inputMat.shape)
#     nRows = inputMat.shape[0]
#     nCols = inputMat.shape[1]
#     S = int(max(nRows, nCols) / 8)
#
#     s2 = int(S / 4)
#
#     for i in range(nRows):
#         y1 = i - s2
#         y2 = i + s2
#
#         if (y1 < 0):
#             y1 = 0
#         if (y2 >= nRows):
#             y2 = nRows - 1
#
#         for j in range(nCols):
#             x1 = j - s2
#             x2 = j + s2
#
#             if (x1 < 0):
#                 x1 = 0
#             if (x2 >= nCols):
#                 x2 = nCols - 1
#             count = (x2 - x1) * (y2 - y1)
#
#             sum = s[y2][x2] - s[y2][x1] - s[y1][x2] + s[y1][x1]
#
#             if ((int)(inputMat[i][j] * count) < (int)(sum * (1.0 - T))):
#                 outputMat[i][j] = 255
#                 # print(i,j)
#             # else:
#             #     outputMat[j][i] = 0
#     return outputMat


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


# def convolution(image, kernel):
#     """
#     This function which takes an image and a kernel and returns the convolution of them.
#     """
#     # Flip the kernel
#     kernel = np.flipud(np.fliplr(kernel))
#     # convolution output filled with zeros
#     output = np.zeros_like(image)
#
#     # Add zero padding to the input image
#     image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
#     image_padded[1:-1, 1:-1] = image
#
#     # Loop over every pixel of the image
#     for x in range(image.shape[0]):
#         for y in range(image.shape[1]):
#             # element-wise multiplication of the kernel and the image
#             output[x, y] = (kernel * image_padded[x: x + 3, y: y + 3]).sum()
#
#     return output


# def threshold(img):
#     rowSize, columnSize = img.shape
#     thresholdedMap = np.zeros((rowSize, columnSize))
#
#     for row in range(1, rowSize - 1):
#         for col in range(1, columnSize - 1):
#
#             # If pixel value is higher than highthreshold, it is strong edge
#             if img[row, col] >= highThresholdRatio:
#                 thresholdedMap[row, col] = strong
#
#             # If pixel value is in between thresholds, it is weak edge
#             elif lowThresholdRatio <= img[row, col] and img[row, col] < highThresholdRatio:
#                 thresholdedMap[row, col] = weak
#
#             # If pixel value is lower than lowthreshold, it is non-relevant pixel, zero out it
#             elif img[row, col] < lowThresholdRatio:
#                 thresholdedMap[row, col] = 0
#
#     return thresholdedMap


# def BlurImage(image):
#     '''
#         This function blurs image with an mean filter
#     :param image:
#     :return:
#     '''
#     mean_kernel = np.array(
#         [
#             [1, 1, 1],
#             [1, 1, 1],
#             [1, 1, 1]
#         ]
#     ) / 9.0
#     im_filtered = np.zeros_like(image)
#     im_filtered[:, :] = convolution(image[:, :], mean_kernel)
#     return im_filtered


def setLabelRec(im, point, label):
    """ Recursive utility function for CCA """
    connectivity = [
        [-1, -1], [-1, 0], [-1, +1],
        [0, -1], [0, +1],
        [1, -1], [1, 0], [1, +1]
    ]  # komsuluk/connectivity matrisi
    shape = im.shape
    im[point[0]][point[1]] = label
    # bu pikseli etiketle
    for offset in connectivity:
        # etraftaki pikseller icin
        y = point[0] + offset[0]  # y noktasi
        x = point[1] + offset[1]  # x noktasi
        if y < 0 or y >= shape[0] or x < 0 or x >= shape[1]:
            continue  # eger goruntunun disina tasildiysa, atla
        if im[y][x] == 255:  # eger cevredeki piksel beyaz ise etiketle
            setLabelRec(im, (y, x), label)  # rekursif cagri


def connectedComponents(im):
    """ Custom connected component analysis """

    label_counter = 0  # etiket sayaci
    white = np.argwhere(im == 255)
    # goruntudeki beyaz olan yerlerin koordinatlarini al
    while len(white) > 0:
        # eger beyaz 100'den fazla beyaz nokta varsa
        setLabelRec(im, white[0], label_counter)
        # beyaz noktayi etiketle, rekursif bicimde
        white = np.argwhere(im == 255)  # beyaz nokta listesini guncelle
        label_counter += 1  # etiket sayacini arttir

    label_counter -= 1  # extract backround
    return label_counter, im


# def applyHysteresisThreshold(img):
#     # Get size of image
#     rowSize = img.shape[0]
#     columnSize = img.shape[1]
#
#     finalImage = np.zeros((rowSize, columnSize))
#
#     # Loop over thresholded map to find weak edges which indeed is strong edge
#     for row in range(1, rowSize - 1):
#         for col in range(1, columnSize - 1):
#
#             if img[row, col] == weak:
#
#                 # Look at 8 neigbours of current pixel to find and connected strong value
#                 if ((img[row + 1, col - 1] == strong) or (img[row + 1, col] == strong) or (
#                         img[row + 1, col + 1] == strong)
#                         or (img[row, col - 1] == strong) or (img[row, col + 1] == strong)
#                         or (img[row - 1, col - 1] == strong) or (img[row - 1, col] == strong) or (
#                                 img[row - 1, col + 1] == strong)):
#                     finalImage[row, col] = 1
#                 else:
#                     finalImage[row, col] = 0
#
#             elif img[row, col] == strong:
#                 finalImage[row, col] = 1
#
#     return finalImage


def dilation(image):
    rows, columns = image.shape
    # print(rows,columns)
    for i in range(rows - 2):
        for j in range(columns - 2):
            # print(i,j)
            if image[i][j] == 255 or image[i][j + 1] == 255 or image[i][j + 2] == 255 or image[i + 1][j] == 255 or \
                    image[i + 1][j + 1] == 255 or image[i + 1][j + 2] == 255 or image[i + 2][j] == 255 or image[i + 2][
                j + 1] == 255 or image[i + 2][j + 2] == 255:
                image[i][j] = 255
            else:
                continue
    return image


def erosion(image):
    rows, columns = image.shape
    # print(rows,columns)
    for i in range(rows - 2):
        for j in range(columns - 2):
            # print(i,j)
            if image[i][j] == 255 and image[i][j + 1] == 255 and image[i][j + 2] == 255 and image[i + 1][j] == 255 and \
                    image[i + 1][j + 1] == 255 and image[i + 1][j + 2] == 255 and image[i + 2][j] == 255 and \
                    image[i + 2][j + 1] == 255 and image[i + 2][j + 2] == 255:
                image[i][j] = 255
            else:
                image[i][j] = 0
                continue
    return image


def opening(image):
    erosion_image = erosion(copy.deepcopy(image))
    dilation_image = dilation(copy.deepcopy(erosion_image))
    resultant_image = copy.deepcopy(dilation_image)
    return resultant_image


def closing(image):
    dilation_image = dilation(copy.deepcopy(image))
    erosion_image = erosion(copy.deepcopy(dilation_image))
    resultant_image = copy.deepcopy(erosion_image)
    return resultant_image


if __name__ == '__main__':
    birdnames = ["bird_1.jpg", "bird_2.jpg", "bird_3.bmp"]
    # birdnames = ["bird_2.jpg"]
    plt.set_cmap(cmap="gray")
    # kernel = np.array([[1, 0, 0, 0, 0],
    #                    [0, 1, 0, 0, 0],
    #                    [0, 0, 1, 0, 0],
    #                    [0, 0, 0, 1, 0],
    #                    [0, 0, 0, 0, 1]], dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    for name in birdnames:
        img = readImage(name)
        img = img_as_ubyte(img)
        if len(img.shape) == 3:
            img = rgb_to_gray(img)

        otsu_threshold = getOtsuThreshold(img)
        thresholded = globalThresholding(img, otsu_threshold)

        close = closing(thresholded)
        open = opening(close)

        labels, result = connectedComponents(open)
        showPlot(result, cmap="nipy_spectral")
        print(name + " NUMBER OF BIRDS: " + str(labels))
        saveOutput(result, name)
