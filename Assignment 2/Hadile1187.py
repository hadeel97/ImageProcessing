import math
import cv2
import numpy as np


def IncreaseContrast(m, n, a1, bb, image):
    r1, g1, b1 = cv2.split(image)
    r = r1.astype(np.int64)
    g = g1.astype(np.int64)
    b = b1.astype(np.int64)

    height, width = r.shape

    tempr = [[0 for _ in range(width)] for _ in range(height)]
    for p in range(height):
        for o in range(width):
            tempr[p][o] = r[p][o]
    tempb = [[0 for _ in range(width)] for _ in range(height)]
    for k in range(height):
        for gr in range(width):
            tempb[k][gr] = b[k][gr]
    tempg = [[0 for _ in range(width)] for _ in range(height)]
    for t in range(height):
        for s in range(width):
            tempg[t][s] = g[t][s]

    tempimgr = [[0 for _ in range(width)] for _ in range(height)]
    for p in range(height):
        for o in range(width):
            tempimgr[p][o] = r[p][o]
    tempimgb = [[0 for _ in range(width)] for _ in range(height)]
    for k in range(height):
        for gr in range(width):
            tempimgb[k][gr] = b[k][gr]
    tempimgg = [[0 for _ in range(width)] for _ in range(height)]
    for t in range(height):
        for s in range(width):
            tempimgg[t][s] = g[t][s]

    tempor = Opening(m, n, tempr)
    tempob = Opening(m, n, tempb)
    tempog = Opening(m, n, tempg)
    tempcr = Closing(m, n, tempimgr)
    tempcb = Closing(m, n, tempimgb)
    tempcg = Closing(m, n, tempimgg)

    imgr = r + (a1 * (r - tempor) - (bb * (tempcr - r)))
    imgg = g + (a1 * (g - tempog) - (bb * (tempcg - g)))
    imgb = b + (a1 * (b - tempob) - (bb * (tempcb - b)))
    merged = cv2.merge([imgr, imgg, imgb])
    merged = merged.astype(np.int64)
    return merged


def Closing(m, n, color):
    kernel = np.ones((m, n))
    mid = math.floor(m / 2)
    clr = erosion(kernel, mid, dilation(kernel, mid, color))

    return clr


def Opening(m, n, color):
    kernel = np.ones((m, n))
    mid = math.floor(m / 2)
    clr1 = dilation(kernel, mid, erosion(kernel, mid, color))

    return clr1


def dilation(kernel, mid, c):
    dilated = [[0 for _ in range(len(c[0]))] for _ in range(len(c))]
    for p in range(len(dilated)):
        for f in range(len(dilated[0])):
            dilated[p][f] = c[p][f]

    for u in range(mid, len(c) - mid):
        for o in range(mid, len(c[0]) - mid):
            temp = getTemp(u, o, kernel, c, mid)
            maxval = temp[0][0]
            for q in range(len(kernel)):
                for v in range(len(kernel[0])):
                    if temp[q][v] <= maxval:
                        dilated[u][o] = maxval
                    else:
                        maxval = temp[q][v]
                        dilated[u][o] = temp[q][v]

    return dilated


def getTemp(i, j, kernell, c, mid):
    temp = np.zeros((len(kernell), len(kernell[0])))
    for x1 in range(len(kernell)):
        for y1 in range(len(kernell[0])):
            temp[x1][y1] = c[i - mid + x1][j - mid + y1]
    return temp


def erosion(kernel, mid, c):
    eroded = [[0 for _ in range(len(c[0]))] for _ in range(len(c))]
    for p in range(len(eroded)):
        for f in range(len(eroded[0])):
            eroded[p][f] = c[p][f]
    for i in range(mid, len(c) - mid):
        for j in range(mid, len(c[0]) - mid):
            temp = getTemp(i, j, kernel, c, mid)
            minval = temp[0][0]
            for m in range(len(kernel)):
                for n in range(len(kernel[0])):
                    if temp[m][n] >= minval:
                        eroded[i][j] = minval
                    else:
                        minval = temp[m][n]
                        eroded[i][j] = temp[m][n]
    return eroded


imgc = cv2.imread("/Users/nano/Desktop/ass2/Suez Canal.png")
imgc = imgc.astype(np.int64)

cv2.imwrite('/Users/nano/Desktop/ass2/Image3311.jpg', IncreaseContrast(3, 3, 1, 1, imgc))
cv2.imwrite('/Users/nano/Desktop/ass2/Image9911.jpg', IncreaseContrast(9, 9, 1, 1, imgc))
cv2.imwrite('/Users/nano/Desktop/ass2/Image3351.jpg', IncreaseContrast(3, 3, 5, 1, imgc))
cv2.imwrite('/Users/nano/Desktop/ass2/Image3315.jpg', IncreaseContrast(3, 3, 1, 5, imgc))

