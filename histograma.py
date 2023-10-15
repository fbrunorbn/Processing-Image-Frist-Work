import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
image = cv2.imread('Fig0320(2)(2nd_from_top).tif', cv2.IMREAD_GRAYSCALE)

# Calcular o histograma
hist = {}
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        val = image[i,j]
        if val in hist:
            hist[val] += 1
        else:
            hist[val] = 1

for i in range (0,256):
    if i in hist:
        continue
    else:
        hist[i] = 0
print(hist)
hist = sorted(hist.items(), key=lambda x: x[0])
print(type(hist))
print(hist)

L,qtd = zip(*hist)

##Agora equalizando!!!!!!!
totalPixel = image.shape[0]*image.shape[1]
print(totalPixel)
porcentagem = []
probAcumulada = []
print(len(hist))
porc = 0
for i in range(0,len(hist)):
    val = hist[i][1]
    porc = porc + (val/totalPixel)
    porcentagem.append(val/totalPixel)
    probAcumulada.append(porc)
print(porcentagem)
print(probAcumulada)

newValues = []
for i in range (0, len(probAcumulada)):
    prob = probAcumulada[i]
    val = prob * 255
    newValues.append(int(val))
print(newValues)

new_image = np.zeros_like(image)
for i in range(0, len(hist)):
    intensidade,_ = hist[i]
    coords = np.argwhere(image == intensidade)
    for coord in coords:
        x,y = coord
        new_image[x,y] = newValues[i]





# Plotar o histograma
plt.bar(L, qtd, width=1.0, color='gray')
plt.title('Histograma em Escala de Cinza')
plt.xlabel('Níveis de Cinza')
plt.ylabel('Frequência')
plt.show()

hist = {}
for i in range(0,new_image.shape[0]):
    for j in range(0,new_image.shape[1]):
        val = new_image[i,j]
        if val in hist:
            hist[val] += 1
        else:
            hist[val] = 1


print(hist)
hist = sorted(hist.items(), key=lambda x: x[0])
print(type(hist))
print(hist)

L,qtd = zip(*hist)

plt.bar(L, qtd, width=1.0, color='gray')
plt.title('Histograma em Escala de Cinza Equalizada')
plt.xlabel('Níveis de Cinza')
plt.ylabel('Frequência')
plt.show()

cv2.imshow('Imagem em Escala de Cinza', image)
cv2.imshow('Imagem em Escala de Cinza Equalizada', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
