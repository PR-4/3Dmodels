
import random
import numpy as np

import pylab as py
import matplotlib.pyplot as plt
import IPython as ipw
from IPython.display import display
from matplotlib import interactive
from matplotlib import style
from matplotlib.widgets import Cursor, Button
from PIL import Image

def horizon(ma):
    
    # funcao de callback:
    def click(event):
        #---- append list with picked values -----:
        x.append(event.xdata)
        y.append(event.ydata)
        print('coordenada Vertical:',y)
        print('coordenada Horizontal:',x)
        plotx.append(event.xdata)
        ploty.append(event.ydata)
            
        #-------------- plot data -------------------: 
        line.set_color('k')
        line.set_marker('o')
        line.set_linestyle('None')
        line.set_data(plotx, ploty)
        
       # ax.figure.canvas.draw()     
        
    # ----- for the case of new clicks -------:
    x = []
    y = []
    
    plotx = []
    ploty = []
   # ----------------- cleaning line object for plotting ------------------:
    fig, ax =plt.subplots()
    line, = ax.plot([],[])
         
    # --------- cursor use for better visualization ------------- :
    ax.set_title("Interpreted Seismic Section")
    cursor=Cursor(ax,horizOn=True,vertOn=True, color='black',useblit=True,linewidth=2.0)
    #ax.style.use(['classic'])  
    plt.style.use(['classic']) #fundo cinza
    plt.imshow(ma)
    plt.grid()

# ------------ Hack because Python 2 doesn't like nonlocal variables that change value -------:
# Lists it doesn't mind.
    picking = [True]
    fig.canvas.mpl_connect('button_press_event', click )
    plt.show()
     
    return x,y
    
    
    
   


def transform(a):
    '''
    Converte a imagem em pixel em real grandeza.
    Entrada(s): imagem (*png), FC da largura, FC da altura
    *FC, fator de conversão
    Saída: matriz com a imagem convertida
    --
    FC = ( real grandeza / 20 ) * 1,2 
    '''
# leitura da imagem do lago:
    imagem_pixel = Image.open(a)
#Conversão de pixel para metros:

#Medidas da imagem original
    largura_pixel = imagem_pixel.size[0]
    altura_pixel = imagem_pixel.size[1]

#Salvando as medidas em uma variável
    medida_px = '{} x {} pixels'.format(largura_pixel, altura_pixel)

#Definindo medida da imagem em metros
    largura_metro =float(input('Entre com a largura da imagem na unidade métrica: '))
    altura_metro = float(input('Entre com a altura da imagem, na unidade métrica: '))

#Salvando as medidas em uma variável
    medida_mt = '{} x {} metros'.format(largura_metro, altura_metro)

#Redimensionando a imagem para medida em metros
    ma = imagem_pixel.resize((int(largura_metro * largura_pixel),int(altura_metro * altura_pixel)))

#Mostrando informações
    print("Informações da imagem original: " + medida_px)
    print("Informações da imagem redimensionada: " + medida_mt)
    
    return ma
    
    
    
    
    
    
