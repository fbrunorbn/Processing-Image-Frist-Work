import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv

img = cv.imread('Fig0646(a)(lenna_original_RGB).tif')
img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
print(img_rgb.shape)