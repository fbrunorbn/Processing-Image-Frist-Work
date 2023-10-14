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
        self.config_Tela()
        self.frames_Menus()
        self.buttons()
        self.combobox()
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
        self.canvaOriginal.place(x=15, y=40)

        self.frameResults = Frame(self.root, background="red")
        self.frameResults.place(relx=0.40, rely=0.03, relheight=0.94, relwidth=0.58)

        self.canvaResultImage = Canvas(self.frameResults, width=640, height=640, bg='#068481')
        self.canvaResultImage.place(x=5, y=8)


    def buttons(self):
        self.bt_loadImage = Button(self.frameAlgs, text="Carregar Imagem", command=self.loadImage)
        self.bt_loadImage.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.05)
        pass

    def loadImage(self):
        self.originalImage = None
        
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.originalImage = cv.imread(self.image_path)
            self.originalImage = cv.cvtColor(self.originalImage, cv.COLOR_BGR2RGB)
            self.drawOriginalImage()

    def drawOriginalImage(self):
        global pilOriginalImage
        pilOriginalImage = Image.fromarray(self.originalImage)
        pilOriginalImage = pilOriginalImage.resize((450,450), Image.Resampling.LANCZOS)
        pilOriginalImage = ImageTk.PhotoImage(pilOriginalImage)
        self.canvaOriginal.create_image(0,0, image=pilOriginalImage, anchor='nw')

    def drawResult(self):
        global pilResultImage

        labelResult = Label(self.canvaResultImage, text="Resultado Negativa", font='arial 15 bold')
        labelResult.place(x=10,y=5)

        pilResultImage = Image.fromarray(self.PILResultImage)
        pilResultImage = pilResultImage.resize((550,550), Image.Resampling.LANCZOS)
        pilResultImage = ImageTk.PhotoImage(pilResultImage)
        self.canvaResultImage.create_image(0,0, image=pilResultImage, anchor='nw')

    def combobox(self):
        listaFunc = ["None","Negativa", "Contraste"]
        labelCombobox = Label(self.frameAlgs, text="Selecione a Funcionalidade")
        labelCombobox.place(x=10,y=500)

        self.comboboxFunc = ttk.Combobox(self.frameAlgs, values=listaFunc)
        self.comboboxFunc.set('None')
        self.comboboxFunc.place(x=10,y=520)
        self.comboboxFunc.bind("<<ComboboxSelected>>", lambda event: self.Functions(str(self.comboboxFunc.get())))

    def Functions(self,value):
        if value == "Negativa":
            negativa = np.zeros_like(self.originalImage)
            height, width, _ = self.originalImage.shape
            for i in range(0, height - 1): 
                for j in range(0, width - 1): 
                    pixel = self.originalImage[i, j]                 
                    pixel[0] = 255 - pixel[0]                     
                    pixel[1] = 255 - pixel[1]                     
                    pixel[2] = 255 - pixel[2]                     
                    negativa[i, j] = pixel
            self.PILResultImage = negativa.copy()
            self.drawResult()

         

        

Algoritmos = Aplication()

