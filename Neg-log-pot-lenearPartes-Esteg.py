import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv
import math

img = cv.imread('Fig0304(a)(breast_digital_Xray).tif')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgGray = (imgGray/255.0)
print(imgGray)

#Negativa
imgNeg = 1.0 - imgGray

#Transformacoes logaritmicas
c = 500 / np.log(1 + np.max(imgGray)) 
img_log = c*(np.log(1 + imgGray))
img_log = np.array(img_log, dtype=np.uint8)

#Correcao de gama
img_gam = np.array(255*(imgGray) ** 5, dtype='uint8')

cv.imshow('Imagem original', img)
#cv.imshow('Imagem Negativa', imgNeg)
#cv.imshow('Imagem Log', img_log)
cv.imshow('Imagem Gamma', img_gam)
cv.waitKey(0)
cv.destroyAllWindows()