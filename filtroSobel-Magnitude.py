import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv

img = cv.imread('Sudoku.png')
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

img_gray = img_gray/255.0

sobelX = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
listaMatrizX = [valor for linha in sobelX for valor in linha]


sobelY = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
listaMatrizY = [valor for linha in sobelY for valor in linha]


def convolucao(img,x,y,tam,M):
    #M = M[::-1]
    ind = 0
    soma = 0
    for i in range (y - tam//2,y + tam//2 +1):
        for j in range (x - tam//2, x + tam//2 +1):
            L = img[i,j]
            soma = soma + L*M[ind]
            ind += 1
    return soma

tamanho = 3
img_sobelX = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
for x in range(1,img_gray.shape[1]-1):
    for y in range(1,img_gray.shape[0]-1):
        #Percorrer toda a imagem
        intensidade = convolucao(img_gray,x,y,tamanho,listaMatrizX)
        img_sobelX[y,x] = intensidade

img_sobelY = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
for x in range(1,img_gray.shape[1]-1):
    for y in range(1,img_gray.shape[0]-1):
        #Percorrer toda a imagem
        intensidade = convolucao(img_gray,x,y,tamanho,listaMatrizY)
        img_sobelY[y,x] = intensidade

#img_gradiente = img_sobelX + img_sobelY
sobX = img_sobelX.copy()
sobY = img_sobelY.copy()

img_sobelX = img_sobelX + abs(img_sobelX.min())
img_sobelX = (img_sobelX/img_sobelX.max())

img_sobelY = img_sobelY + abs(img_sobelY.min())
img_sobelY = (img_sobelY/img_sobelY.max())

img_gradiente = np.sqrt(img_sobelX**2 + img_sobelY**2)

# Normalizar o gradiente para a faixa de 0-255
img_gradiente = (img_gradiente / img_gradiente.max())

img_gradiente_NNorm = np.sqrt(sobX**2 + sobY**2)
img_gradiente_NNorm = (img_gradiente_NNorm / img_gradiente_NNorm.max())
print(sobX.max())
print(sobY.min())



cv.imshow('Original', img_gray)
cv.imshow('Sobel X', img_sobelX)
cv.imshow('Sobel Y', img_sobelY)
cv.imshow('Gradiente', img_gradiente)
cv.imshow('Gradiente Nao Normalizado', img_gradiente_NNorm)
cv.waitKey(0)
cv.destroyAllWindows()