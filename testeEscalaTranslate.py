import matplotlib.pyplot as plt
import numpy as np
import string
import cv2
import math

img = cv2.imread('Fig0236(a)(letter_T).tif')  # Carregar a imagem em tons de cinza

print(img.shape)


T_s = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
angle_degrees = 10  # Ângulo de rotação em graus
angle_radians = np.radians(angle_degrees)  # Converter para radianos

T_r = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 1]])

T = T_s @ T_r

#T = T_s

print(T)

# Calcula as dimensões da imagem transformada
output_shape = (int(img.shape[0]), int(img.shape[1]), 3)
img_transformed = np.zeros(output_shape, dtype=np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pixel_value = img[i, j]
        input_coords = np.array([i, j, 1])#Criação da matriz homogenea da imagem original
        transformed_coords = np.dot(T, input_coords)#aplica a matriz da tranf afim
        transformed_coords = np.round(transformed_coords / transformed_coords[2])  # Arredondar para índices inteiros, e normaliza pelo valor da terceira posição, para voltar a ser 1 da homogenea
        i_out, j_out, _ = transformed_coords.astype(int)

        if 0 <= i_out < output_shape[0] and 0 <= j_out < output_shape[1]:
            #verifica se o pixel da transformada esta dentro do limite da img_transformed, que sera apresentada
        #if 0 <= i_out < img.shape[0] and 0 <= j_out < img.shape[1]:
            img_transformed[i_out, j_out, :] = pixel_value


# Função para interpolação do vizinho mais próximo
T_inv = np.linalg.inv(T)

def nearest_neighbors(i, j, M, T_inv):
    #i,j -> coordenadas da transformada a ser transformada de volta para a original
    x_max, y_max = M.shape[0] - 1, M.shape[1] - 1 #shape da original
    x, y, _ = T_inv @ np.array([i, j, 1])#coord da original mapeada
    if np.floor(x) == x and np.floor(y) == y:#verifica se o ja estao em um pixel inteiro da original
        x, y = int(x), int(y)
        return M[x, y]
    if np.abs(np.floor(x) - x) < np.abs(np.ceil(x) - x):#verifica se é o piso ou o teto a ser pego
        x = int(np.floor(x))
    else:
        x = int(np.ceil(x))
    if np.abs(np.floor(y) - y) < np.abs(np.ceil(y) - y):#verifica se é o piso ou o teto a ser pego
        y = int(np.floor(y))
    else:
        y = int(np.ceil(y))
    #verifica se o x ou o y estão dentro da imagem original
    if x > x_max:
        x = x_max
    if y > y_max:
        y = y_max
    return M[x, y]

img_nn = np.zeros(output_shape, dtype=np.uint8)

for i, row in enumerate(img_nn):
    for j, col in enumerate(row):
        img_nn[i, j] = nearest_neighbors(i, j, img, T_inv)

def bilinear_interpolation(i, j, M, T_inv):
    x_max, y_max = M.shape[0] - 1, M.shape[1] - 1
    x, y, _ = T_inv @ np.array([i, j, 1])

    x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
    x_ceil, y_ceil = int(np.ceil(x)), int(np.ceil(y))
    #pegando as coordenadas vizinhas (antes e dps)

    # Verifica se as coordenadas transformadas estão dentro dos limites da imagem original
    if 0 <= x_floor < x_max and 0 <= x_ceil < x_max and 0 <= y_floor < y_max and 0 <= y_ceil < y_max:
        # Quadrado que engloba o ponto x, y achado pela transformação
        q11 = M[x_floor, y_floor]
        q12 = M[x_floor, y_ceil]
        q21 = M[x_ceil, y_floor]
        q22 = M[x_ceil, y_ceil]

        # Distancia dos pontos achados para os cantos da caixa
        dx = x - x_floor
        dy = y - y_floor

        # Aplica a fórmula da interpolação bilinear
        interpolated_value = (1 - dx) * (1 - dy) * q11 + dx * (1 - dy) * q21 + (1 - dx) * dy * q12 + dx * dy * q22

        return interpolated_value
    else:
        return 0  # Retorna 0 se estiver fora dos limites da imagem original

img_int = np.zeros(output_shape, dtype=np.uint8)

for i, row in enumerate(img_int):
    for j, col in enumerate(row):
        img_int[i, j] = bilinear_interpolation(i, j, img, T_inv)

cv2.imshow('Imagem original', img_transformed)
cv2.imshow('Imagem near', img_nn)
cv2.imshow('Imagem bilinear', img_int)
cv2.waitKey(0)
cv2.destroyAllWindows()
