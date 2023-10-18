import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv
import math

img = cv.imread('Fig0320(2)(2nd_from_top).tif')
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
img_gam = np.array(255*(imgGray) ** 2, dtype='uint8')

#Linear Def por partes - Contrast stretching
def pixelVal(pix, r1, s1, r2, s2): 
	if (0 <= pix and pix <= r1): 
		return (s1 / r1)*pix 
	elif (r1 < pix and pix <= r2): 
		return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
	else: 
		return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 

imgGray = imgGray*255.0
img_gray_stret = np.zeros((int(imgGray.shape[0]), int(imgGray.shape[1])))

# Define parameters. 
r1 = 20
s1 = 0
r2 = 200
s2 = 255

for x in range(0,imgGray.shape[1]):
	for y in range (0,imgGray.shape[0]):
		img_gray_stret[y,x] = pixelVal(imgGray[y,x],r1,s1,r2,s2)

print(img_gray_stret.max())
print(img_gray_stret.min())

pixelVal_vec = np.vectorize(pixelVal) 
  
# Apply contrast stretching. 
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)



cv.imshow('Imagem original', img)
cv.imshow('Imagem Negativa', imgNeg)
cv.imshow('Imagem Log', img_log)
cv.imshow('Imagem Gamma', img_gam)
cv.imshow('Linear por partes - Contraste Stretching',img_gray_stret)
cv.imshow('2',contrast_stretched)
cv.waitKey(0)
cv.destroyAllWindows()