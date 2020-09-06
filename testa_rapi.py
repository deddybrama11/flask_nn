import imageio
import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.stats import kurtosis, skew, entropy
from statistics import mean

f = csv.writer(open('coba.csv', 'w', newline=""))
onlyfiles = [f for f in listdir(r'tomatlokal\sehat') if isfile(join(r'tomatlokal\sehat', f))]
for gambar in onlyfiles:
    gambar = r'\%s' % (gambar)
    # cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(r'tomatlokal\sehat' + gambar, cv2.IMREAD_UNCHANGED)

    width = 200
    height = 200
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    #background removal
    # mask = np.zeros(img.shape[:2], np.uint8)
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # rect = (3, 3, 1000, 1000)
    # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img = img * mask2[:, :, np.newaxis]

    x = np.array(img)
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]

    R = 0
    G = 0
    B = 0
    std_red = 0
    std_green = 0
    std_blue = 0

    R = np.mean(r)
    G = np.mean(g)
    B = np.mean(b)
    std_red = np.std(r)
    std_green = np.std(g)
    std_blue = np.std(b)

    row = [std_red, std_green, std_blue,  kurtosis(x, axis=None), skew(x, axis=None), entropy(x, axis=None), 'Sehat']

    mean = 0
    for image in x:
        mean += np.mean(image, axis=None)
    mean = mean / len(x)

    f.writerow(row)

onlyfiles = [f for f in listdir(r'tomatlokal\septoria') if isfile(join(r'tomatlokal\septoria', f))]
for gambar in onlyfiles:
    gambar = r'\%s' % (gambar)

    img = cv2.imread(r'tomatlokal\septoria' + gambar, cv2.IMREAD_UNCHANGED)

    width = 200
    height = 200
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # background removal
    # mask = np.zeros(img.shape[:2], np.uint8)
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # rect = (3, 3, 1000, 1000)
    # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img = img * mask2[:, :, np.newaxis]

    x = np.array(img)
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]

    R = 0
    G = 0
    B = 0
    std_red = 0
    std_green = 0
    std_blue = 0

    R = np.mean(r)
    G = np.mean(g)
    B = np.mean(b)
    std_red = np.std(r)
    std_green = np.std(g)
    std_blue = np.std(b)

    row = [std_red, std_green, std_blue,  kurtosis(x, axis=None), skew(x, axis=None), entropy(x, axis=None), 'Septoria']

    mean = 0
    for image in x:
        mean += np.mean(image, axis=None)
    mean = mean / len(x)

    f.writerow(row)

onlyfiles = [f for f in listdir(r'tomatlokal\yellow leaf curl') if isfile(join(r'tomatlokal\yellow leaf curl', f))]
for gambar in onlyfiles:
    gambar = r'\%s' % (gambar)

    img = cv2.imread(r'tomatlokal\yellow leaf curl' + gambar, cv2.IMREAD_UNCHANGED)
    print(img.dtype)

    width = 200
    height = 200
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # background removal
    # mask = np.zeros(img.shape[:2], np.uint8)
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # rect = (3, 3, 1000, 1000)
    # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img = img * mask2[:, :, np.newaxis]

    x = np.array(img)
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]

    R = 0
    G = 0
    B = 0
    std_red = 0
    std_green = 0
    std_blue = 0

    R = np.mean(r)
    G = np.mean(g)
    B = np.mean(b)
    std_red = np.std(r)
    std_green = np.std(g)
    std_blue = np.std(b)

    row = [std_red, std_green, std_blue,  kurtosis(x, axis=None), skew(x, axis=None), entropy(x, axis=None),
           'Yellow Leaf Curl']

    mean = 0
    for image in x:
        mean += np.mean(image, axis=None)
    mean = mean / len(x)

    f.writerow(row)


def imgProcessing(gambar):
    img = gambar

    width = 200
    height = 200
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # background removal
    # mask = np.zeros(img.shape[:2], np.uint8)
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    # rect = (3, 3, 1000, 1000)
    # cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img = img * mask2[:, :, np.newaxis]

    x = np.array(img)
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]



    R = np.mean(r)
    G = np.mean(g)
    B = np.mean(b)
    std_red = np.std(r)
    std_green = np.std(g)
    std_blue = np.std(b)

    row = [std_red, std_green, std_blue,  kurtosis(x, axis=None), skew(x, axis=None), entropy(x, axis=None),'Undetected']

    mean = 0
    for image in x:
        mean += np.mean(image, axis=None)
    mean = mean / len(x)



    return row

