import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv

img = cv.imread('Fig0308(a)(fractured_spine).tif')
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#Limiarizacao
limiar = 200
img_limiar = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])), dtype=np.uint8)

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


for i in range(0,img_gray.shape[0]):
    for j in range(0,img_gray.shape[1]):
        val = img_gray[i,j]
        if val>=limiar:
            img_limiar[i,j] = 255

#Convolucao


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
img = cv.imread('Fig0335(a)(ckt_board_saltpep_prob_pt05).tif')
img_gray_med = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

img_gray_med = img_gray_med/255.0

def mediana(img,x,y,tam):
    med = []
    for i in range (y - tam//2,y + tam//2 +1):
        for j in range (x - tam//2, x + tam//2 +1):
            if (0 <= j < img.shape[1]) and (0<= i < img.shape[0]):
                L = img[i,j]
                med.append(L)
            else:
                continue
    median = np.array(med)
    intensidade = np.median(median)
    return intensidade

tamanho = 5

img_mediana = np.zeros((int(img_gray_med.shape[0]),int(img_gray_med.shape[1])))
for x in range(0,img_gray_med.shape[1]):
    for y in range(0,img_gray_med.shape[0]):
        #Percorrer toda a imagem
        intensidade = mediana(img_gray_med,x,y,tamanho)
        img_mediana[y,x] = intensidade



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

img_nitidezLapla = img_gray + (0.9*img_lapla)

#Filtro High Boost
tamanho = 3
matriz = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]])
matriz = matriz/16.0
listaMatriz = [valor for linha in matriz for valor in linha]

img_gauss = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
for x in range(0,img_gray.shape[1]):
    for y in range(0,img_gray.shape[0]):
        #Percorrer toda a imagem
        intensidade = convolucao(img_gray,x,y,tamanho,listaMatriz)
        img_gauss[y,x] = intensidade

img_bordas_highBust = img_gray - img_gauss
img_highBust = img_gray + (0.9*img_bordas_highBust)



cv.imshow('Imagem Gamma', img_limiar)
cv.imshow('Filtro Generico',  img_generico)
cv.imshow('Filtro Simples',  img_simples)
cv.imshow('Filtro Ponderada',  img_pond)
cv.imshow('Filtro Laplaciano', img_lapla)
cv.imshow('Filtro Realçe Laplaciano', img_nitidezLapla)
cv.imshow('Original', img_gray)
cv.imshow('Filtro da Mediana', img_mediana)
cv.imshow('Filtro Gauss',img_gauss)
cv.imshow('BORDA High Bust', img_bordas_highBust)
cv.imshow('High Bust', img_highBust)
cv.waitKey(0)
cv.destroyAllWindows()
