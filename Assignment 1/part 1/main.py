import math
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io

# opening the image from desktop and reading it
image = Image.open("/Users/hadilekhalifa/Desktop/Suez Canal.png")
width, height = image.size

# thedirectory where i will save the output images
directory = r'/Users/hadilekhalifa/Desktop/imageproject'
os.chdir(directory)

# first array for cityblock matrix
citybd = [[np.inf for x in range(width)] for y in range(height)]
citybd[310][175] = 0
citybd[150][200] = 0

# first array for euclidean matrix
eucd = [[np.inf for x in range(width)] for y in range(height)]
eucd[150][200] = 0
eucd[310][175] = 0

# first array for chessboard matrix
chessboardd = [[np.inf for x in range(width)] for y in range(height)]
chessboardd[150][200] = 0
chessboardd[310][175] = 0

# setting one point to zero to measure the distance between the big ship and the small one
distcity = [[np.inf for x in range(width)] for y in range(height)]
distcity[150][200] = 0

disteuc = [[np.inf for x in range(width)] for y in range(height)]
disteuc[150][200] = 0

distchess = [[np.inf for x in range(width)] for y in range(height)]
distchess[150][200] = 0


# measuring city block distance function
def cityblock(x, y, j, k):
    dis = abs(x - j) + abs(y - k)
    return dis


# measuring euclidean distance function
def euclidean(x, y, j, k):
    dis = int(math.sqrt(math.pow(x - j, 2) + math.pow(y - k, 2)))
    return dis


# measuring chessboard distance function
def chessboard(x, y, j, k):
    dis = max(abs(x - j), abs(y - k))
    return dis


# a function that takes two points and measures and returns the distance using the type it gets
def distance(x, y, j, k, distance_type):
    if distance_type == "cityblock":
        d = cityblock(x, y, j, k)

    if distance_type == "euclidean":
        d = euclidean(x, y, j, k)

    if distance_type == "chessboard":
        d = chessboard(x, y, j, k)
    return d


h = len(citybd)
w = len(citybd[0])


# first pass algorithm cityblock algorithm
def cityblockmatrixtransform(h, w, cbd):
    for x in range(1, h - 1):
        for y in range(1, w):
            if cbd[x][y] == 0:
                cbd[x][y] = 0
            else:
                d1 = cbd[x - 1][y - 1] + distance(x, y, x - 1, y - 1, "cityblock")
                d2 = cbd[x - 1][y] + distance(x, y, x - 1, y, "cityblock")
                d3 = cbd[x][y - 1] + distance(x, y, x, y - 1, "cityblock")
                d4 = cbd[x + 1][y - 1] + distance(x, y, x + 1, y - 1, "cityblock")
                cbd[x][y] = min(min(min(d1, d2), d3), d4)

    array = np.array(cbd)
    filename = "Suez_1_city.bmp"
    cv2.imwrite(filename, array)


    # city block second pass algorithm
    for x in range(h - 2, 0, -1):
        for y in range(w - 2, -1, -1):
            if cbd[x][y] == 0:
                cbd[x][y] = 0
            else:
                d1 = cbd[x + 1][y + 1] + distance(x, y, x + 1, y + 1, "cityblock")
                d2 = cbd[x + 1][y] + distance(x, y, x + 1, y, "cityblock")
                d3 = cbd[x][y + 1] + distance(x, y, x, y + 1, "cityblock")
                d4 = cbd[x - 1][y + 1] + distance(x, y, x - 1, y + 1, "cityblock")
                cbd[x][y] = min(min(min(d1, d2), d3), d4)

    array = np.array(cbd)
    filename = "Suez_final_city.bmp"
    cv2.imwrite(filename, array)
    return cbd


# first pass using euclidean algorithm
def eucmatrixtransform(h, w, eucdistance):
    for x in range(1, h - 1):
        for y in range(1, w):
            if eucdistance[x][y] == 0:
                eucdistance[x][y] = 0
            else:
                d1 = eucdistance[x - 1][y - 1] + distance(x, y, x - 1, y - 1, "euclidean")
                d2 = eucdistance[x - 1][y] + distance(x, y, x - 1, y, "euclidean")
                d3 = eucdistance[x][y - 1] + distance(x, y, x, y - 1, "euclidean")
                d4 = eucdistance[x + 1][y - 1] + distance(x, y, x + 1, y - 1, "euclidean")
                if eucdistance[x][y] > min(min(min(d1, d2), d3), d4):
                    eucdistance[x][y] = min(min(min(d1, d2), d3), d4)

    array = np.array(eucdistance)
    filename = "Suez_1_Euclidean.bmp"
    cv2.imwrite(filename, array)

    # euclidean second pass algorithm
    for x in range(h - 2, 0, -1):
        for y in range(w - 2, -1, -1):
            if eucdistance[x][y] == 0:
                eucdistance[x][y] = 0
            else:
                d1 = eucdistance[x + 1][y + 1] + distance(x, y, x + 1, y + 1, "euclidean")
                d2 = eucdistance[x + 1][y] + distance(x, y, x + 1, y, "euclidean")
                d3 = eucdistance[x][y + 1] + distance(x, y, x, y + 1, "euclidean")
                d4 = eucdistance[x - 1][y + 1] + distance(x, y, x - 1, y + 1, "euclidean")
                if eucdistance[x][y] > min(min(min(d1, d2), d3), d4):
                    eucdistance[x][y] = min(min(min(d1, d2), d3), d4)

    array = np.array(eucdistance)
    filename = "Suez_final_Euclidean.bmp"
    cv2.imwrite(filename, array)
    return eucdistance


# first pass using chessboard algorithm
def chessboardmatrixtransform(h, w, chessbd):
    for x in range(1, h - 1):
        for y in range(1, w):
            if chessbd[x][y] == 0:
                chessbd[x][y] = 0
            else:
                d1 = chessbd[x - 1][y - 1] + distance(x, y, x - 1, y - 1, "chessboard")
                d2 = chessbd[x - 1][y] + distance(x, y, x - 1, y, "chessboard")
                d3 = chessbd[x][y - 1] + distance(x, y, x, y - 1, "chessboard")
                d4 = chessbd[x + 1][y - 1] + distance(x, y, x + 1, y - 1, "chessboard")
                chessbd[x][y] = min(min(min(d1, d2), d3), d4)

    array = np.array(chessbd, dtype=np.float_)
    filename = "Suez_1_Chess.bmp"
    cv2.imwrite(filename, array)

    # chessboard second pass
    for x in range(h - 2, 0, -1):
        for y in range(w - 2, -1, -1):
            if chessbd[x][y] == 0:
                chessbd[x][y] = 0
            else:
                d1 = chessbd[x + 1][y + 1] + distance(x, y, x + 1, y + 1, "chessboard")
                d2 = chessbd[x + 1][y] + distance(x, y, x + 1, y, "chessboard")
                d3 = chessbd[x][y + 1] + distance(x, y, x, y + 1, "chessboard")
                d4 = chessbd[x - 1][y + 1] + distance(x, y, x - 1, y + 1, "chessboard")
                chessbd[x][y] = min(min(min(d1, d2), d3), d4)

    array = np.array(chessbd, dtype=np.float_)
    filename = "Suez_final_Chess.bmp"
    cv2.imwrite(filename, array)
    return chessbd


# calling the three functions to get the euc matrix, cityblock matrix and chessboard matrix
eucmatrixtransform(h, w, eucd)
cityblockmatrixtransform(h, w, citybd)
chessboardmatrixtransform(h, w, chessboardd)

# getting the distance between the small ship and the large ship using the implemented functions

d1 = distance(310, 175, 150, 200, "euclidean")
d2 = distance(310, 175, 150, 200, "cityblock")
d3 = distance(310, 175, 150, 200, "chessboard")

# distance from ships to the bank
image = Image.open(r"/Users/hadilekhalifa/Desktop/Suez Canal.png")
width, height = image.size

img = cv2.imread(r"/Users/hadilekhalifa/Desktop/Suez Canal.png", 0)
edges = cv2.Canny(img, 240, 255)
indices = np.where(edges != [0])
coordinates = list(zip(indices[0], indices[1]))
# from this I noticed the edge points of the ships and the banks
print(coordinates)

bigshipedges = [161, 194, 161, 214]
smallshipedges = [301, 174, 301, 177]
bankA = [161, 150, 161, 294]
bankB = [301, 133, 301, 289]

leftdistancebig = [[np.inf for x in range(width)] for y in range(height)]
leftdistancebig[161][194] = 0
rightdistancebig = [[np.inf for x in range(width)] for y in range(height)]
rightdistancebig[161][214] = 0
leftdistancesmall = [[np.inf for x in range(width)] for y in range(height)]
leftdistancesmall[301][174] = 0
rightdistancesmall = [[np.inf for x in range(width)] for y in range(height)]
rightdistancesmall[301][177] = 0


# first pass algorithm cityblock algorithm
def cityblockmatrixtransform1(cbd):
    for x in range(1, height - 1):
        for y in range(1, width):
            if cbd[x][y] == 0:
                cbd[x][y] = 0
            else:
                d1 = cbd[x - 1][y - 1] + distance(x, y, x - 1, y - 1, "cityblock")
                d2 = cbd[x - 1][y] + distance(x, y, x - 1, y, "cityblock")
                d3 = cbd[x][y - 1] + distance(x, y, x, y - 1, "cityblock")
                d4 = cbd[x + 1][y - 1] + distance(x, y, x + 1, y - 1, "cityblock")
                cbd[x][y] = min(min(min(d1, d2), d3), d4)

    # city block second pass algorithm
    for x in range(height - 2, 0, -1):
        for y in range(width - 2, -1, -1):
            if cbd[x][y] == 0:
                cbd[x][y] = 0
            else:
                d1 = cbd[x + 1][y + 1] + distance(x, y, x + 1, y + 1, "cityblock")
                d2 = cbd[x + 1][y] + distance(x, y, x + 1, y, "cityblock")
                d3 = cbd[x][y + 1] + distance(x, y, x, y + 1, "cityblock")
                d4 = cbd[x - 1][y + 1] + distance(x, y, x - 1, y + 1, "cityblock")
                cbd[x][y] = min(min(min(d1, d2), d3), d4)
    return cbd


d = cityblockmatrixtransform1(leftdistancebig)
d1 = cityblockmatrixtransform1(rightdistancebig)

d2 = cityblockmatrixtransform1(leftdistancesmall)
d3 = cityblockmatrixtransform1(rightdistancesmall)

print("the distances from big ship to left bank:", d[161][150])
print("the distances from big ship to right bank:", d1[161][294])

print("the distance from small ship to left bank:", d2[301][133])
print("the distance from small ship to right bank:", d3[301][289])
