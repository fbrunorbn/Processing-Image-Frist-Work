from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
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
            #self.originalImage = cv.cvtColor(self.originalImage, cv.COLOR_BGR2RGB)
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

        labelResult = Label(self.canvaResultImage, text=text, font='arial 15 bold')
        labelResult.place(x=10,y=5)

        pilResultImage = Image.fromarray(imagemResultado)
        pilResultImage = pilResultImage.resize((640,640), Image.Resampling.LANCZOS)
        pilResultImage = ImageTk.PhotoImage(pilResultImage)
        self.canvaResultImage.create_image(0,0, image=pilResultImage, anchor='nw')

    def combobox(self):
        listaFunc = ["None", "Negativa", "Transformações Logaritmicas", "Correção de gama (Potência)", "Histograma Equalizado", "Limiarização (Binarização)", "Filtro Genérico por Convolução", "Escala e Rotacao com Interpolção Linear", "Escala e Rotação com Interpolação do Vizinho"]
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
        elif value == "Histograma Equalizado":
            self.HistEqual()
            self.drawResult()

    def Negativa(self):
        imgGray = cv.cvtColor(self.originalImage, cv.COLOR_BGR2GRAY)
        imgGray = (imgGray/255.0)
        imgNeg = 1.0 - imgGray
        imgNeg = imgNeg*255
        self.PILResultImage = imgNeg.copy()
        self.resultados.append(("Negativa",self.PILResultImage))
        self.entryInfo.delete(0, "end")

    def TransfLog(self):
        #Expande os pixels escuros da imagem em comparação com valores de pixel mais altos.
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
        imgGray = cv.cvtColor(self.originalImage, cv.COLOR_BGR2GRAY)
        pot = self.entryInfo.get()
        imgGray = (imgGray/255.0)
        img_gam = np.array(255*(imgGray) ** 5, dtype='uint8')
        if pot != "":
            img_gam = np.array(255*(imgGray) ** int(pot), dtype='uint8')
        self.PILResultImage = img_gam.copy()
        self.resultados.append(("Correção de Gama",self.PILResultImage))
        self.entryInfo.delete(0, "end")
    
    def HistEqual(self):
        pass
         

        

Algoritmos = Aplication()

