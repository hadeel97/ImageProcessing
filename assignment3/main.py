from PIL import Image

img = Image.open('/Users/hadilekhalifa/Desktop/assignment3/Camera2.jpg')
dict1 = [[0 for _ in range(2)] for _ in range(256)]

for m in range(len(dict1)):
    dict1[m] = [m, m]

d = open("/Users/hadilekhalifa/Desktop/assignment3/Dict.txt", "w+")
LZW = open("/Users/hadilekhalifa/Desktop/assignment3/LZWCode.txt", "w+")
CR = open("/Users/hadilekhalifa/Desktop/assignment3/CompRatio.txt", "w+")


def imageComp(image: Image, dict: list):
    pixels = list(image.getdata())
    dict, output = getOutput(dict, pixels)
    CRatio = GetCompressionRatio(output, pixels)
    LZW.write(str(output))
    d.write(str(dict))
    CR.write(str(CRatio))
    return output


def CheckInDict(dict: list, pixel: int):
    flag = False
    for j in range(len(dict)):
        if pixel == dict[j][1]:
            flag = True
    return flag


def getpixel(pixel, dict):
    for re in range(len(dict)):
        if dict[re][1] == pixel:
            return dict[re][0]


def getOutput(dict, pixels1):
    output1 = ""
    m = 0
    for j in range(len(pixels1) - 1):
        pixel = pixels1[m]
        flag = CheckInDict(dict, pixel)
        while flag:
            output = getpixel(pixel, dict)
            if m < len(pixels1) - 1:
                m += 1
                pixel = int(str(pixel) + str(pixels1[m]))
                flag = CheckInDict(dict, pixel)
            else:
                break
        else:
            output1 += str(output) + " "
            dict = AddToDict(dict, pixel)

    return dict, output1


def AddToDict(dict: list, input):
    dict.append([len(dict), input])
    return dict


def GetCompressionRatio(output, pixels):
    out = output.split(" ")
    Or = ""
    O = ""
    for m in range(len(out)-1):
        out[m] = int(float(out[m]))
    for i in range(len(pixels)):
        Or += str(format(pixels[i], 'b'))
    for j in range(len(out)-1):
        O += str(format(int(out[j]), 'b'))
    CRatio = (len(Or) / len(O))

    return CRatio


imageComp(img, dict1)
