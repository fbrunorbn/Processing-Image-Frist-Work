import matplotlib.pyplot as plt
import numpy as np
import string
import cv2 as cv

img = cv.imread('Fig0646(a)(lenna_original_RGB).tif')
img_rgb = img.copy()
hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

print(img_rgb.shape)

#RGBtoHSV

def RGBtoHSV(r,g,b):
    r,g,b = r/255.0, g/255.0, b/255.0

    #calculando minimos
    cmax = max(r,g,b)
    cmin = min(r,g,b)
    dif  = cmax - cmin

    #Calculando o Hue
    if cmax == cmin:
        hue = 0
    elif cmax == r:
        hue = (60 *((g - b)/dif)+360)%360 #Aplicando a formula
    elif cmax == g:
        hue = (60 *((b - r)/dif)+120)%360
    elif cmax == b:
        hue = (60 *((r - g)/dif)+240)%360

    #Calculando saturation
    if cmax == 0:
        sat = 0
    else:
        sat = (dif/cmax)*255

    #Calculando value
    val = cmax*255

    return [hue/2,sat,val]



img_hsv = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.float32)
img_gray_simples = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
img_gray_ponderada = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
img_negativa = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.uint8)
for x in range(0,img_rgb.shape[1]):
    for y in range(0,img_rgb.shape[0]):
        r,g,b = img_rgb[y,x]
        valor = (float(r) + float(g) + float(b)) / 3
        img_gray_simples[y, x] = valor
        img_gray_ponderada[y,x] = 0.299*r+0.587*g+0.114*b

        img_hsv[y,x] = RGBtoHSV(r,g,b)

        r_neg = 255 - r
        g_neg = 255 - g
        b_neg = 255 - b
        img_negativa[y,x] = [r_neg,g_neg,b_neg]

pixels_manual = img_hsv[0:2, 0:2]

# Acessar os primeiros 4 pixels de hsv_img
pixels_opencv = hsv_img[0:2, 0:2]

# Exibir os valores de Hue (H), Saturation (S) e Value (V) para os primeiros 4 pixels
for i in range(2):
    for j in range(2):
        h_manual, s_manual, v_manual = pixels_manual[i, j]
        h_opencv, s_opencv, v_opencv = pixels_opencv[i, j]
        print(f"Pixel ({i},{j}) - Manual (H, S, V): ({h_manual}, {s_manual}, {v_manual})")
        print(f"Pixel ({i},{j}) - OpenCV (H, S, V): ({h_opencv}, {s_opencv}, {v_opencv})")

img_hsv = img_hsv.astype(np.uint8)
cv.imshow('Original', img_rgb)
cv.imshow('HSV', img_hsv)
cv.imshow('HSV OPENCV', hsv_img)
cv.imshow('Escala cinza Simples', img_gray_simples)
cv.imshow('Escala cinza Ponderada', img_gray_ponderada)
cv.imshow('Colorida Negativa', img_negativa)
cv.waitKey(0)
cv.destroyAllWindows()