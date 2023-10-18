from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from matplotlib import _cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk()

class Aplication():
    def __init__(self):
        self.root = root
        self.originalImage = None
        self.PILResultImage = None
        self.resultados = []
        self.indiceResultados = -1
        self.combSelect = ""
        self.config_Tela()
        self.frames_Menus()
        self.buttons()
        self.combobox()
        self.entradas()
        root.mainloop()

    def config_Tela(self):
        self.root.title("Algoritmos de Processamento de Imagens")
        self.root.configure(background="white")
        self.root.geometry("1300x700+0+0")
        self.root.resizable(False,False)

    def frames_Menus(self):
        self.frameAlgs = Frame(self.root, background="grey")
        self.frameAlgs.place(relx=0.02, rely=0.03, relheight=0.94, relwidth=0.37)

        self.canvaOriginal = Canvas(self.frameAlgs, width=450, height=450, bg='#068481')
        self.canvaOriginal.grid(row=6, column=0,columnspan=2, sticky="ew")

        self.frameResults = Frame(self.root, background="red")
        self.frameResults.place(relx=0.40, rely=0.03, relheight=0.94, relwidth=0.58)

        self.canvaResultImage = Canvas(self.frameResults, width=640, height=640, bg='#068481')
        self.canvaResultImage.place(x=5, y=8)


    def buttons(self):
        self.bt_loadImage = Button(self.frameAlgs, text="Carregar Imagem", command=self.loadImage, width=70, height=2)
        self.bt_loadImage.grid(row=0, column=0, columnspan=2, sticky="ew")
        pass
        self.bt_Applie = Button(self.frameAlgs, text="Aplicar Alg", command=self.Functions)
        #self.bt_Applie = Button(self.frameAlgs, text="Aplicar Alg")
        self.bt_Applie.grid(row=5, column=1, sticky="ew")
        
        self.bt_nextImage = Button(self.frameResults, text="Proxima Imagem", command=self.NextImage, width=13,height=1)
        self.bt_nextImage.place(x=651,y=300)
        self.bt_prevImage = Button(self.frameResults, text="Imagem Anterior", command=self.PrevImage, width=13,height=1)
        self.bt_prevImage.place(x=651,y=328)
    
    def entradas(self):
        labelEntrada = Label(self.frameAlgs, text="Digite a entrada", width=25, height=1)
        labelEntrada.grid(row=4, column=0, sticky="ew")

        self.entryInfo = Entry(self.frameAlgs)
        self.entryInfo.grid(row=5, column=0, sticky="ew")

    def loadImage(self):
        self.originalImage = None
        self.resultados.clear()
        self.entryInfo.delete(0, "end")
        
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.originalImage = cv.imread(self.image_path)
            self.drawOriginalImage()
    
    def NextImage(self):
        self.indiceResultados +=1
        if self.indiceResultados >= len(self.resultados):
            self.indiceResultados = 0
        self.drawResult()

    def PrevImage(self):
        self.indiceResultados -=1
        if self.indiceResultados < 0:
            self.indiceResultados = len(self.resultados) - 1
        self.drawResult()

    def drawOriginalImage(self):
        global pilOriginalImage
        pilOriginalImage = Image.fromarray(self.originalImage)
        pilOriginalImage = pilOriginalImage.resize((450,450), Image.Resampling.LANCZOS)
        pilOriginalImage = ImageTk.PhotoImage(pilOriginalImage)
        self.canvaOriginal.create_image(0,0, image=pilOriginalImage, anchor='nw')

    def drawResult(self):
        global pilResultImage

        text = self.resultados[self.indiceResultados][0]
        imagemResultado = self.resultados[self.indiceResultados][1]

        labelResult = Label(self.canvaResultImage, text=text, font='arial 15 bold', width=50, height=1)
        labelResult.place(x=10,y=5)

        pilResultImage = Image.fromarray(imagemResultado)
        pilResultImage = pilResultImage.resize((640,640), Image.Resampling.LANCZOS)
        pilResultImage = ImageTk.PhotoImage(pilResultImage)
        self.canvaResultImage.create_image(0,0, image=pilResultImage, anchor='nw')

    def combobox(self):
        listaFunc = ["None", "Negativa", "Transformações Logaritmicas", "Correção de gama (Potência)", "Linear definida por partes", "Histograma e Histograma Equalizado", "Limiarização (Binarização)", "Filtro Genérico por Convolução","Filtro de Suavização da média simples", "Filtro de Suavização da média ponderada", "Filtro da Mediana", "Aguçamento por Laplaciano", "Aguçamento por High-Boost", "Filtros de Sobel X e Y", "Detecção não linear de bordas pelo gradiente", "Imagem RGB para HSV", "Escala de Cinza por media simples", "Escala de Cinza por media ponderada", "Negativo de Colorida", "Escala e Rotacao com Interpolção Linear", "Escala e Rotação com Interpolação do Vizinho"]
        labelCombobox = Label(self.frameAlgs, text="Selecione a Funcionalidade",width=25, height=1)
        labelCombobox.grid(row=1, column=0, sticky="ew")

        self.comboboxFunc = ttk.Combobox(self.frameAlgs, values=listaFunc)
        self.comboboxFunc.set('None')
        self.comboboxFunc.grid(row=2, column=0, sticky="ew")

        #self.comboboxFunc.bind("<<ComboboxSelected>>", lambda event: self.ComboboxSelect(self.comboboxFunc.get()))
        self.comboboxFunc.bind("<<ComboboxSelected>>", lambda event: self.ComboboxSelect(str(self.comboboxFunc.get())))
    
    def ComboboxSelect(self,value):
        self.combSelect = value
        print(self.combSelect)

    def Functions(self):
        value = self.combSelect
        print(value)
        if value == "Negativa":
            self.Negativa()
            self.drawResult()
        elif value == "Transformações Logaritmicas":
            self.TransfLog()
            self.drawResult()
        elif value == "Correção de gama (Potência)":
            self.CorrecGama()
            self.drawResult()
        elif value == "Linear definida por partes":
            self.LinearPartes()
            self.drawResult()
        elif value == "Histograma e Histograma Equalizado":
            self.HistEqual()
            self.drawResult()
        elif value == "Limiarização (Binarização)":
            self.Limiar()
            self.drawResult()
        elif value == "Filtro Genérico por Convolução":
            self.Generico()
            self.drawResult()
        elif value == "Filtro de Suavização da média simples":
            self.SuaveSimples()
            self.drawResult()
        elif value == "Filtro de Suavização da média ponderada":
            self.SuavePonderada()
            self.drawResult()
        elif value == "Filtro da Mediana":
            self.Mediana()
            self.drawResult()
        elif value == "Aguçamento por Laplaciano":
            self.Laplaciano()
            self.drawResult()
        elif value == "Aguçamento por High-Boost":
            self.HighBoost()
            self.drawResult()
        elif value == "Filtros de Sobel X e Y":
            self.Sobel()
            self.drawResult()
        elif value == "Detecção não linear de bordas pelo gradiente":
            self.Gradiente()
            self.drawResult()
        elif value == "Imagem RGB para HSV":
            self.RGBtoHSV()
            self.drawResult()
        elif value == "Escala de Cinza por media simples":
            self.CinzaSimples()
            self.drawResult()
        elif value == "Escala de Cinza por media ponderada":
            self.CinzaPonderada()
            self.drawResult()
        elif value == "Negativo de Colorida":
            self.NegativaCor()
            self.drawResult()
        elif value == "Escala e Rotacao com Interpolção Linear":
            self.EscRotIntLinear()
        elif value == "Escala e Rotação com Interpolação do Vizinho":
            self.EscRotIntVizinho()

    def Negativa(self):
        self.resultados.clear()
        self.indiceResultados = -1
        imgGray = cv.cvtColor(self.originalImage, cv.COLOR_BGR2GRAY)
        imgGray = (imgGray/255.0)
        imgNeg = 1.0 - imgGray
        imgNeg = imgNeg*255
        self.PILResultImage = imgNeg.copy()
        self.resultados.append(("Negativa",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def TransfLog(self):
        #Expande os pixels escuros da imagem em comparação com valores de pixel mais altos., ajuda a realçar detalhes em baixa luminosidade
        self.resultados.clear()
        self.indiceResultados = -1
        info = self.entryInfo.get()
        imgGray = cv.cvtColor(self.originalImage, cv.COLOR_BGR2GRAY)
        c = 255 / np.log(1 + np.max(imgGray))
        if info != "":
            c = float(info)
        img_log = c*(np.log(1 + imgGray))
        img_log = np.array(img_log, dtype=np.uint8)
        self.PILResultImage = img_log.copy()
        self.resultados.append(("Transformação Logaritmica",self.PILResultImage))
        self.entryInfo.delete(0, "end")
    
    def CorrecGama(self):
        #Ajudar o brilho e contraste da imagem
        self.resultados.clear()
        self.indiceResultados = -1
        imgGray = cv.cvtColor(self.originalImage, cv.COLOR_BGR2GRAY)
        pot = self.entryInfo.get()
        imgGray = (imgGray/255.0)
        img_gam = np.array(255*(imgGray) ** 5, dtype='uint8')
        if pot != "":
            img_gam = np.array(255*(imgGray) ** float(pot), dtype='uint8')
        self.PILResultImage = img_gam.copy()
        self.resultados.append(("Correção de Gama",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def LinearPartes(self):
        self.resultados.clear()
        self.indiceResultados = -1
        def pixelVal(pix, r1, s1, r2, s2): 
            if (0 <= pix and pix <= r1): 
                return (s1 / r1)*pix 
            elif (r1 < pix and pix <= r2): 
                return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
            else: 
                return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 
        # processo de processamento de imagem que visa aumentar o contraste entre os tons de uma imagem, esticando a faixa de valores dos pixels para que ela cubra um intervalo mais amplo. Essa técnica é frequentemente usada para realçar detalhes em uma imagem, tornando as áreas mais escuras mais escuras e as áreas mais claras mais claras.
        imgGray = cv.cvtColor(self.originalImage, cv.COLOR_BGR2GRAY)
        img_gray_stret = np.zeros((int(imgGray.shape[0]), int(imgGray.shape[1])), dtype=np.uint8)
        parametros = self.entryInfo.get()
        if parametros != "":
            parametros = (parametros).split(" ")
            r1 = float(parametros[0])
            s1 = float(parametros[1])
            r2 = float(parametros[2])
            s2 = float(parametros[3])
        else:
            r1 = 70
            s1 = 0
            r2 = 220
            s2 = 255
        for x in range(0,imgGray.shape[1]):
            for y in range (0,imgGray.shape[0]):
                img_gray_stret[y,x] = pixelVal(imgGray[y,x],r1,s1,r2,s2)

        self.PILResultImage = img_gray_stret.copy()
        self.resultados.append(("Linear definida por partes",self.PILResultImage))
        self.entryInfo.delete(0, "end")
    
    def HistEqual(self):
        self.resultados.clear()
        self.indiceResultados = -1
        image = cv.cvtColor(self.originalImage, cv.COLOR_BGR2GRAY)

        # Calcular o histograma

        #Inicia com a contagem de intensidade dos pixels
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
        hist = sorted(hist.items(), key=lambda x: x[0])

        L,qtd = zip(*hist)#Pegando uma lista de intensidade e qtd pixels tem dela

        #Iniciando a equalização

        #Calculando a porcentagem dos pixels de intensidade e a porcentagem acumulada
        totalPixel = image.shape[0]*image.shape[1]
        porcentagem = []
        probAcumulada = []
        porc = 0
        for i in range(0,len(hist)):
            val = hist[i][1]
            porc = porc + (val/totalPixel)
            porcentagem.append(val/totalPixel)
            probAcumulada.append(porc)

        #Determinando os novos valores das intensidades
        newValues = []
        for i in range (0, len(probAcumulada)):
            prob = probAcumulada[i]
            val = prob * 255
            newValues.append(int(val))

        #Criando a nova imagem com os novos valores de intensidades
        new_image = np.zeros_like(image)
        for i in range(0, len(hist)):
            intensidade,_ = hist[i]
            coords = np.argwhere(image == intensidade)
            for coord in coords:
                x,y = coord
                new_image[x,y] = newValues[i]

        #Plotando
        fig,ax = plt.subplots()
        ax.bar(L, qtd, width=1.0, color='gray')
        #plt.bar()
        ax.set_title('Histograma em Escala de Cinza')
        ax.set_xlabel('Níveis de Cinza')
        ax.set_ylabel('Frequência')
        fig.savefig("hist.png")
        plt.close(fig)

        img = cv.imread('hist.png')

        os.remove('hist.png')

        self.PILResultImage = img.copy()
        self.resultados.append(("Histograma",self.PILResultImage))

        self.PILResultImage = new_image.copy()
        self.resultados.append(("Imagem Equalizada",self.PILResultImage))

        #Criando o histograma da equalizada
        hist = {}
        for i in range(0,new_image.shape[0]):
            for j in range(0,new_image.shape[1]):
                val = new_image[i,j]
                if val in hist:
                    hist[val] += 1
                else:
                    hist[val] = 1

        hist = sorted(hist.items(), key=lambda x: x[0])

        L,qtd = zip(*hist)

        fig,ax = plt.subplots()
        ax.bar(L, qtd, width=1.0, color='gray')
        #plt.bar()
        ax.set_title('Histograma Equaliado em Escala de Cinza')
        ax.set_xlabel('Níveis de Cinza')
        ax.set_ylabel('Frequência')
        fig.savefig("hist_Eq.png")
        plt.close(fig)

        img = cv.imread('hist_Eq.png')

        os.remove('hist_Eq.png')

        self.PILResultImage = img.copy()
        self.resultados.append(("Histograma Equalizado",self.PILResultImage))

        self.entryInfo.delete(0, "end")

    def Limiar(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)

        #Limiarizacao
        limiar = self.entryInfo.get()
        if limiar != "":
            limiar = int(limiar)
        else:
            limiar = 200
        img_limiar = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])), dtype=np.uint8)
        for i in range(0,img_gray.shape[0]):
            for j in range(0,img_gray.shape[1]):
                val = img_gray[i,j]
                if val>=limiar:
                    img_limiar[i,j] = 255

        self.PILResultImage = img_limiar.copy()
        self.resultados.append(("Binarizacao por Limiar",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def convolucao(self,img,x,y,tam,M):
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
    
    def Generico(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
        tamanho = self.entryInfo.get()
        if tamanho != "":
            tamanho = int(tamanho)
        else:
            tamanho = 3
        matriz = np.random.randint(-10,10,(tamanho,tamanho))
        listaMatriz = [valor for linha in matriz for valor in linha]


        img_generico = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
        for x in range(0,img_gray.shape[1]):
            for y in range(0,img_gray.shape[0]):
                #Percorrer toda a imagem
                intensidade = self.convolucao(img_gray,x,y,tamanho,listaMatriz)
                img_generico[y,x] = intensidade
        #Normalizando a imagem
        img_generico = img_generico + np.min(abs(img_generico))
        img_generico = img_generico/np.max(img_generico)
        img_generico = img_generico*255
        self.PILResultImage = img_generico.copy()
        self.resultados.append((f"Filtro Genérico {tamanho}x{tamanho}",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def SuaveSimples(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
        tamanho = self.entryInfo.get()
        if tamanho != "":
            tamanho = int(tamanho)
        else:
            tamanho = 3
        matriz = np.ones((tamanho,tamanho))
        matriz = matriz / (tamanho**2) #normalizada
        listaMatriz = [valor for linha in matriz for valor in linha]


        img_simples = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
        for x in range(0,img_gray.shape[1]):
            for y in range(0,img_gray.shape[0]):
                #Percorrer toda a imagem
                intensidade = self.convolucao(img_gray,x,y,tamanho,listaMatriz)
                img_simples[y,x] = intensidade

        img_simples = img_simples/np.max(img_simples)
        img_simples = img_simples*255

        self.PILResultImage = img_simples.copy()
        self.resultados.append((f"Filtro de Suavização da media simples {tamanho}x{tamanho}",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def SuavePonderada(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
        tamanho = self.entryInfo.get()
        if tamanho != "":
            tamanho = int(tamanho)
        else:
            tamanho = 3

        matriz = np.random.randint(0,10,(tamanho,tamanho))
        soma = np.sum(matriz)
        matriz = matriz / soma #normalizada
        listaMatriz = [valor for linha in matriz for valor in linha]


        img_pond = np.zeros((int(img_gray.shape[0]), int(img_gray.shape[1])))
        for x in range(0,img_gray.shape[1]):
            for y in range(0,img_gray.shape[0]):
                #Percorrer toda a imagem
                intensidade = self.convolucao(img_gray,x,y,tamanho,listaMatriz)
                img_pond[y,x] = intensidade

        img_pond = img_pond/np.max(img_pond)
        img_pond = img_pond*255

        self.PILResultImage = img_pond.copy()
        self.resultados.append((f"Filtro de Suavização da media ponderada {tamanho}x{tamanho}",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def Mediana(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
        tamanho = self.entryInfo.get()
        if tamanho != "":
            tamanho = int(tamanho)
        else:
            tamanho = 3

        img_gray = img_gray/255.0

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

        img_mediana = np.zeros((int(img_gray.shape[0]),int(img_gray.shape[1])))
        for x in range(0,img_gray.shape[1]):
            for y in range(0,img_gray.shape[0]):
                #Percorrer toda a imagem
                intensidade = mediana(img_gray,x,y,tamanho)
                img_mediana[y,x] = intensidade
        
        img_mediana = img_mediana*255

        self.PILResultImage = img_mediana.copy()
        self.resultados.append((f"Filtro da Mediana {tamanho}x{tamanho}",self.PILResultImage))
        self.entryInfo.delete(0, "end")
    
    def Laplaciano(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
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
                intensidade = self.convolucao(img_gray,x,y,tamanho,listaMatriz)
                img_lapla[y,x] = intensidade

        img_nitidezLapla = img_gray + (0.9*img_lapla)
        img_nitidezLapla = img_nitidezLapla*255

        self.PILResultImage = (img_gray*255).copy()
        self.resultados.append(("Imagem Original",self.PILResultImage))
        self.PILResultImage = (img_lapla*255).copy()
        self.resultados.append(("Filtro Laplaciano",self.PILResultImage))
        self.PILResultImage = (img_nitidezLapla).copy()
        self.resultados.append(("Aguçamento por Laplaciano",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def HighBoost(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
        img_gray = img_gray/255.0
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
                intensidade = self.convolucao(img_gray,x,y,tamanho,listaMatriz)
                img_gauss[y,x] = intensidade

        img_bordas_highBust = img_gray - img_gauss
        img_highBust = img_gray + (0.9*img_bordas_highBust)
        img_highBust = img_highBust*255

        self.PILResultImage = (img_gray*255).copy()
        self.resultados.append(("Imagem Original",self.PILResultImage))
        self.PILResultImage = (img_bordas_highBust*255).copy()
        self.resultados.append(("Filtro High-Boost",self.PILResultImage))
        self.PILResultImage = (img_highBust).copy()
        self.resultados.append(("Aguçamento por High-Boost",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def Sobel(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
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

        img_sobelX = img_sobelX + abs(img_sobelX.min())
        img_sobelX = (img_sobelX/img_sobelX.max())

        img_sobelY = img_sobelY + abs(img_sobelY.min())
        img_sobelY = (img_sobelY/img_sobelY.max())

        self.PILResultImage = (img_gray*255).copy()
        self.resultados.append(("Imagem Original",self.PILResultImage))
        self.PILResultImage = (img_sobelX*255).copy()
        self.resultados.append(("Filtro Sobel X Normalizado",self.PILResultImage))
        self.PILResultImage = (img_sobelY*255).copy()
        self.resultados.append(("Filtro Sobel Y Normalizado",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def Gradiente(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_gray = cv.cvtColor(self.originalImage,cv.COLOR_BGR2GRAY)
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

        self.PILResultImage = (img_gray*255).copy()
        self.resultados.append(("Imagem Original",self.PILResultImage))
        self.PILResultImage = (img_gradiente_NNorm*255).copy()
        self.resultados.append(("Filtro Gradiente Não Normalizado",self.PILResultImage))
        self.PILResultImage = (img_gradiente*255).copy()
        self.resultados.append(("Filtro Gradiente Normalizado",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def RGBtoHSV(self):
        self.resultados.clear()
        self.indiceResultados = -1

        def toHSV(r,g,b):
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
        
        img_rgb = self.originalImage

        img_hsv = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.float32)
        for x in range(0,img_rgb.shape[1]):
            for y in range(0,img_rgb.shape[0]):
                r,g,b = img_rgb[y,x]
                img_hsv[y,x] = toHSV(r,g,b)
        
        img_hsv = np.uint8(img_hsv)

        self.PILResultImage = (cv.cvtColor(img_rgb,cv.COLOR_BGR2RGB)).copy()
        self.resultados.append(("Imagem original",self.PILResultImage))
        self.PILResultImage = (img_hsv).copy()
        self.resultados.append(("Imagem em HSV",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def CinzaSimples(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_rgb = self.originalImage
        img_gray_simples = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        for x in range(0,img_rgb.shape[1]):
            for y in range(0,img_rgb.shape[0]):
                r,g,b = img_rgb[y,x]
                valor = (float(r) + float(g) + float(b)) / 3
                img_gray_simples[y, x] = valor
        self.PILResultImage = (cv.cvtColor(img_rgb,cv.COLOR_BGR2RGB)).copy()
        self.resultados.append(("Imagem original",self.PILResultImage))
        self.PILResultImage = (img_gray_simples).copy()
        self.resultados.append(("Imagem em Escala de Cinza Simples",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def CinzaPonderada(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_rgb = self.originalImage
        img_gray_ponderada = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        for x in range(0,img_rgb.shape[1]):
            for y in range(0,img_rgb.shape[0]):
                r,g,b = img_rgb[y,x]
                img_gray_ponderada[y,x] = 0.299*r+0.587*g+0.114*b

        self.PILResultImage = (cv.cvtColor(img_rgb,cv.COLOR_BGR2RGB)).copy()
        self.resultados.append(("Imagem original",self.PILResultImage))
        self.PILResultImage = (img_gray_ponderada).copy()
        self.resultados.append(("Imagem em Escala de Cinza Ponderada",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def NegativaCor(self):
        self.resultados.clear()
        self.indiceResultados = -1
        img_rgb = cv.cvtColor(self.originalImage,cv.COLOR_BGR2RGB)
        img_negativa = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.uint8)
        for x in range(0,img_rgb.shape[1]):
            for y in range(0,img_rgb.shape[0]):
                r,g,b = img_rgb[y,x]
                r_neg = 255 - r
                g_neg = 255 - g
                b_neg = 255 - b
                img_negativa[y,x] = [r_neg,g_neg,b_neg]

        self.PILResultImage = (img_rgb).copy()
        self.resultados.append(("Imagem original",self.PILResultImage))
        self.PILResultImage = (img_negativa).copy()
        self.resultados.append(("Negativo da Imagem",self.PILResultImage))
        self.entryInfo.delete(0, "end")
    
    def EscRotIntLinear(self):
        global pilResultImage
        self.resultados.clear()
        self.indiceResultados = -1
        img = self.originalImage
        infos = self.entryInfo.get()
        if infos != "":
            infos = infos.split(" ")
            escala = float(infos[0])
            angulo = int(infos[1])
        else:
            escala = 1.5
            angulo = 10

        T_s = np.array([[escala, 0, 0], [0, escala, 0], [0, 0, 1]])
        angle_degrees = angulo  # Ângulo de rotação em graus
        angle_radians = np.radians(angle_degrees)  # Converter para radianos

        T_r = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                        [np.sin(angle_radians), np.cos(angle_radians), 0],
                        [0, 0, 1]])

        T = T_s @ T_r

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

        self.PILResultImage = (img_nn).copy()
        self.resultados.append((f"Imagem escalada {escala} e rotacionada {angulo} com interp Linear",self.PILResultImage))
        self.entryInfo.delete(0, "end")

        

        text = self.resultados[self.indiceResultados][0]
        imagemResultado = self.resultados[self.indiceResultados][1]

        labelResult = Label(self.canvaResultImage, text=text, font='arial 15 bold', width=50, height=1)
        labelResult.place(x=10,y=5)

        pilResultImage = Image.fromarray(imagemResultado)
        pilResultImage = ImageTk.PhotoImage(pilResultImage)
        self.canvaResultImage.create_image(0,0, image=pilResultImage, anchor='nw')

    def EscRotIntVizinho(self):
        global pilResultImage
        self.resultados.clear()
        self.indiceResultados = -1
        img = self.originalImage
        infos = self.entryInfo.get()
        if infos != "":
            infos = infos.split(" ")
            escala = float(infos[0])
            angulo = int(infos[1])
        else:
            escala = 1.5
            angulo = 10

        T_s = np.array([[escala, 0, 0], [0, escala, 0], [0, 0, 1]])
        angle_degrees = angulo  # Ângulo de rotação em graus
        angle_radians = np.radians(angle_degrees)  # Converter para radianos

        T_r = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                        [np.sin(angle_radians), np.cos(angle_radians), 0],
                        [0, 0, 1]])

        T = T_s @ T_r

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

        self.PILResultImage = (img_int).copy()
        self.resultados.append((f"Imagem escalada {escala} e rotacionada {angulo} com interp Vizinho",self.PILResultImage))
        self.entryInfo.delete(0, "end")

        

        text = self.resultados[self.indiceResultados][0]
        imagemResultado = self.resultados[self.indiceResultados][1]

        labelResult = Label(self.canvaResultImage, text=text, font='arial 15 bold', width=50, height=1)
        labelResult.place(x=10,y=5)

        pilResultImage = Image.fromarray(imagemResultado)
        pilResultImage = ImageTk.PhotoImage(pilResultImage)
        self.canvaResultImage.create_image(0,0, image=pilResultImage, anchor='nw')




Algoritmos = Aplication()

