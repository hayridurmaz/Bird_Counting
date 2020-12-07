import copy
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

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


def setLabelRec(im, point, label):
    """ Recursive utility function for CCA """
    connectivity = [
        [-1, -1], [-1, 0], [-1, +1],
        [0, -1], [0, +1],
        [1, -1], [1, 0], [1, +1]
    ]  # connectivity matrix
    shape = im.shape
    im[point[0]][point[1]] = label
    # tag the pixel
    for offset in connectivity:
        # 8N neighbors
        y = point[0] + offset[0]  # y
        x = point[1] + offset[1]  # x
        if y < 0 or y >= shape[0] or x < 0 or x >= shape[1]:
            continue  # edges
        if im[y][x] == 255:  # neighbors white; same label
            setLabelRec(im, (y, x), label)


def connectedComponents(im):
    """ Custom connected component analysis """
    label_counter = 0
    white = np.argwhere(im == 255)
    # select white pixels
    while len(white) > 0:
        setLabelRec(im, white[0], label_counter)
        white = np.argwhere(im == 255)  # update white list
        label_counter += 1  # increase counter

    return label_counter, im


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


def opening(image, iteration=1):
    for i in range(iteration):
        erosion_image = erosion(copy.deepcopy(image))
        dilation_image = dilation(copy.deepcopy(erosion_image))
        resultant_image = copy.deepcopy(dilation_image)
    return resultant_image


def closing(image, iteration=1):
    for i in range(iteration):
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
        name = name.replace("bmp", "png")
        saveOutput(img, name)
        plt.set_cmap(cmap="gray")

        img = img_as_ubyte(img)
        if len(img.shape) == 3:
            img = rgb_to_gray(img)
        saveOutput(img, "gray_" + name)
        otsu_threshold = getOtsuThreshold(img)
        thresholded = globalThresholding(img, otsu_threshold)
        saveOutput(thresholded, "thresholded_" + name)

        close = closing(thresholded)
        saveOutput(close, "closed_" + name)

        open = opening(close)
        saveOutput(open, "opened_" + name)

        # dilated = dilation(open)

        labels, result = connectedComponents(open)
        showPlot(result, cmap="nipy_spectral")
        print(name + " NUMBER OF BIRDS: " + str(labels))
        saveOutput(result, "labels_" + name)
