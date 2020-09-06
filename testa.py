import imageio
import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.stats import kurtosis, skew, entropy
from statistics import mean

def imgProcessing(gambar):
    img = gambar

    width = 200
    height = 200
    dim = (width, height)

    pic = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    x = np.array(pic)
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]


    std_red = np.std(r)
    std_green = np.std(g)
    std_blue = np.std(b)

    row = [std_red, std_green, std_blue,  kurtosis(x, axis=None), skew(x, axis=None), entropy(x, axis=None),3]

    mean = 0
    for image in x:
        mean += np.mean(image, axis=None)
    mean = mean / len(x)



    return row
