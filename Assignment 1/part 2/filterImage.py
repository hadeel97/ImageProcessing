import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_file = r'/Users/hadilekhalifa/Desktop/guc.jpg'
img = cv2.imread(img_file, 0)
# thedirectory where i will save the output images
directory = r'/Users/hadilekhalifa/Desktop/task2'
os.chdir(directory)


def filterimg(image, type, cutoff, order):
    imgarr = np.fft.fftshift(np.fft.fft2(image))
    height = len(imgarr)
    width = len(imgarr[0])
    array = [[0 for x in range(len(imgarr[0]))] for y in range(len(imgarr))]
    for i in range(len(imgarr)):
        for j in range(len(imgarr[0])):
            array[i][j] = math.sqrt(pow(i - height / 2, 2) + pow(j - width / 2, 2))
    if type == "idealfilter":
        imat = [[0 for x in range(len(imgarr[0]))] for y in range(len(imgarr))]
        for i in range(len(imat)):
            for j in range(len(imat[0])):
                if array[i][j] <= cutoff:
                    imat[i][j] = 1
                else:
                    imat[i][j] = 0
        for m in range(len(img) - 1):
            for n in range(len(img[0]) - 1):
                imgarr[m][n] = imgarr[m][n] * imat[m][n]
    if type == "butterworthfilter":
        bmat = [[0 for x in range(len(imgarr[0]))] for y in range(len(imgarr))]
        for i in range(len(bmat)):
            for j in range(len(bmat[0])):
                bmat[i][j] = 1 / (1 + pow((array[i][j] / cutoff), 2 * order))
        for x in range(len(imgarr)):
            for y in range(len(imgarr[0])):
                imgarr[x][y] = imgarr[x][y] * bmat[x][y]
    if type == "guassian":
        gmat = [[0 for x in range(len(imgarr[0]))] for y in range(len(imgarr))]
        for i in range(len(gmat)):
            for j in range(len(gmat[0])):
                gmat[i][j] = math.exp(-pow(array[i][j], 2) / (2 * pow(cutoff, 2)))
        for x in range(len(imgarr)):
            for y in range(len(imgarr[0])):
                imgarr[x][y] = imgarr[x][y] * gmat[x][y]
    imgarr = np.fft.ifft2(np.fft.ifftshift(imgarr))
    return imgarr


GUC_ILPF_5 = filterimg(img, "idealfilter", 5, 0)
GUC_ILPF_30 = filterimg(img, "idealfilter", 30, 0)
GUC_ILPF_50 = filterimg(img, "idealfilter", 50, 0)
GUC_BLPF_5 = filterimg(img, "butterworthfilter", 5, 0)
GUC_BLPF_30 = filterimg(img, "butterworthfilter", 30, 0)
GUC_BLPF_50 = filterimg(img, "butterworthfilter", 50, 0)
GUC_GLPF_5 = filterimg(img, "guassian", 5, 0)
GUC_GLPF_30 = filterimg(img, "guassian", 30, 0)
GUC_GLPF_50 = filterimg(img, "guassian", 50, 0)

x1 = np.real(GUC_ILPF_5)
plt.imshow(x1, cmap="gray")
plt.imsave("GUC_ILPF_5.jpg", x1, cmap="gray")

x2 = np.real(GUC_ILPF_30)
plt.imshow(x2, cmap="gray")
plt.imsave("GUC_ILPF_30.jpg", x2, cmap="gray")

x3 = np.real(GUC_ILPF_50)
plt.imshow(x3, cmap="gray")
plt.imsave("GUC_ILPF_50.jpg", x3, cmap="gray")

x4 = np.real(GUC_BLPF_5)
plt.imshow(x4, cmap="gray")
plt.imsave("GUC_BLPF_5.jpg", x4, cmap="gray")

x5 = np.real(GUC_BLPF_30)
plt.imshow(x5, cmap="gray")
plt.imsave("GUC_BLPF_30.jpg", x5, cmap="gray")

x6 = np.real(GUC_BLPF_50)
plt.imshow(x6, cmap="gray")
plt.imsave("GUC_BLPF_50.jpg", x6, cmap="gray")

x7 = np.real(GUC_GLPF_5)
plt.imshow(x7, cmap="gray")
plt.imsave("GUC_GLPF_5.jpg", x7, cmap="gray")

x8 = np.real(GUC_GLPF_30)
plt.imshow(x8, cmap="gray")
plt.imsave("GUC_GLPF_30.jpg", x8, cmap="gray")

x9 = np.real(GUC_GLPF_50)
plt.imshow(x9, cmap="gray")
plt.imsave("GUC_GLPF_50.jpg", x9, cmap="gray")