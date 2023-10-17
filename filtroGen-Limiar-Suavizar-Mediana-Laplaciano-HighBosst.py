import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv

img = cv.imread('Fig0308(a)(fractured_spine).tif')
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#Limiarizacao
limiar = 200
img_limiar = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])), dtype=np.uint8)

for i in range(0,img_gray.shape[0]):
    for j in range(0,img_gray.shape[1]):
        val = img_gray[i,j]
        if val>=limiar:
            img_limiar[i,j] = 255

#Convolucao
def convolucao(img,x,y,tam,M):
    M = M[::-1]
    ind = 0
    soma = 0
    for i in range (y - tam//2,y + tam//2 +1):
        for j in range (x - tam//2, x + tam//2 +1):
            if (0 <= j < img.shape[1]) and (0<= i < img.shape[0]):
                L = img[i,j]
                soma = soma + L*M[ind]
                ind += 1
            else:
                soma = soma + 0*M[ind]
                ind += 1
    return soma

#Filto Genérico por convolucao
tamanho = 5
matriz = np.random.randint(-10,10,(tamanho,tamanho))
soma = np.sum(matriz)
#matriz = matriz / soma #normalizada
print(matriz)
listaMatriz = [valor for linha in matriz for valor in linha]


img_generico = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
for x in range(0,img_gray.shape[1]):
    for y in range(0,img_gray.shape[0]):
        #Percorrer toda a imagem
        intensidade = convolucao(img_gray,x,y,tamanho,listaMatriz)
        img_generico[y,x] = intensidade
print(np.max(img_generico))
print(np.min(img_generico))
print(np.min(abs(img_generico)))
img_generico = img_generico + np.min(abs(img_generico))
img_generico = img_generico/np.max(img_generico)
print(np.max(img_generico))

#Filto suavizacao simples
matriz = np.ones((tamanho,tamanho))
matriz = matriz / (tamanho**2) #normalizada
print(matriz)
listaMatriz = [valor for linha in matriz for valor in linha]


img_simples = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
for x in range(0,img_gray.shape[1]):
    for y in range(0,img_gray.shape[0]):
        #Percorrer toda a imagem
        intensidade = convolucao(img_gray,x,y,tamanho,listaMatriz)
        img_simples[y,x] = intensidade

img_simples = img_simples/np.max(img_simples)

#Filto suavizacao ponderada
matriz = np.random.randint(0,10,(tamanho,tamanho))
soma = np.sum(matriz)
matriz = matriz / soma #normalizada
print(matriz)
listaMatriz = [valor for linha in matriz for valor in linha]


img_pond = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
for x in range(0,img_gray.shape[1]):
    for y in range(0,img_gray.shape[0]):
        #Percorrer toda a imagem
        intensidade = convolucao(img_gray,x,y,tamanho,listaMatriz)
        img_pond[y,x] = intensidade

img_pond = img_pond/np.max(img_pond)

#Fitro da mediana




#Filtro Laplaciano
tamanho = 3
matriz = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])
listaMatriz = [valor for linha in matriz for valor in linha]

img_gray = img_gray/255.0

img_lapla = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
for x in range(0,img_gray.shape[1]):
    for y in range(0,img_gray.shape[0]):
        #Percorrer toda a imagem
        intensidade = convolucao(img_gray,x,y,tamanho,listaMatriz)
        img_lapla[y,x] = intensidade

im_lap = cv.Laplacian(img_gray, -1, (3,3))
print(np.max(im_lap))
print(np.max(img_gray))
print(np.max(img_lapla))
img_nitidezLapla = img_gray + (0.1*im_lap)



cv.imshow('Imagem Gamma', img_limiar)
#cv.imshow('Filtro Generico',  img_generico)
#cv.imshow('Filtro Simples',  img_simples)
#cv.imshow('Filtro Ponderada',  img_pond)
cv.imshow('Filtro Laplaciano', img_lapla)
cv.imshow('Filtro Realçe Laplaciano', img_nitidezLapla)
cv.imshow('Teste', im_lap)
cv.imshow('Original', img_gray)
cv.waitKey(0)
cv.destroyAllWindows()
