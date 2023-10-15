import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv

img = cv.imread('Fig0359(a)(headCT_Vandy).tif')
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#Limiarizacao
limiar = 200
img_limiar = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])), dtype=np.uint8)

for i in range(0,img_gray.shape[0]):
    for j in range(0,img_gray.shape[1]):
        val = img_gray[i,j]
        if val>=limiar:
            img_limiar[i,j] = 255

#Filto Genérico por convolucao
tamanho = 3
matriz = np.random.rand(tamanho,tamanho)
soma = np.sum(matriz)
matriz = matriz / soma #normalizada

img_generico = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])), dtype=np.uint8)

for x in range(0,img_gray.shape[0]):
    for y in range(0,img_gray.shape[1]):
        reg = []
        for i in range(y-tamanho//2,y+tamanho//2+1):
            for j in range(x-tamanho//2,x+tamanho//2+1):
                if (i >= 0 and i <= img_gray.shape[1]) and (j>=0 and j<= img_gray.shape[0]):
                    reg.append(img_gray[i,j])


cv.imshow('Imagem Gamma', img_limiar)
cv.waitKey(0)
cv.destroyAllWindows()
