import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import projections, style

style.use('ggplot')

def f(x1,x2):
    return  x1**2+3*x2**2

def grafico_3d():
 x = np.linspace(-100,100,100)
 y = np.linspace(-100,100,100)

 X,Y = np.meshgrid(x,y)

 Z = f(X,Y)

 fig, ax =plt.subplots(subplot_kw=dict(projection='3d'))                #grafico 3d
 ax.plot_surface(X,Y,Z, rstride=5, cstride=5, cmap='plasma')

def curvas_de_nivel():

 x = np.linspace(-15,15,100)
 y = np.linspace(-15,15,100)

 X,Y = np.meshgrid(x,y)

 Z = f(X,Y)

 fig, ax =plt.subplots()                                                 #curvas de nível + gradiente
 contour = ax.contour(X,Y,Z)
 plt.clabel(contour)
 #ax.imshow(Z,extent=(-10,10,-10,10),cmap='plasma')
 plt.plot(px,py)

x=-10
y=10
cx=cy=10
it=0

px = np.array([])
py = np.array([])
img = np.array([])
iteracao = np.array([])


x1=sp.Symbol('x1')
x2=sp.Symbol('x2')
alpha=sp.Symbol('alpha')

funcao = x1**2+3*x2**2


while((cx)>0.000001) or ((cy)>0.00001):      
 px = np.append(px, x)
 py = np.append(py, y)
 
 gradx1 = sp.diff(funcao, x1)
 gradx2 = sp.diff(funcao, x2)

 gradx1 = gradx1.subs(x1, x).subs(x2,y)
 gradx2 = gradx2.subs(x1, x).subs(x2, y)

 vfuncao = funcao.subs(x1,x).subs(x2,y)
 img = np.append(img, vfuncao)

 alfax1 = (x-alpha*gradx1)
 alfax2 = (y-alpha*gradx2)

 exp = funcao.subs(x1, alfax1).subs(x2, alfax2)

 derivada = sp.diff(exp, alpha)

 raiz = sp.solve(derivada)
 raiz = raiz[0]
 
 pontox1 = alfax1.subs(alpha, raiz)
 pontox2 = alfax2.subs(alpha, raiz)
 
 print('it : ',it,' Ponto x1: ', float(pontox1), '  Ponto x2:', float(pontox2))

 cx = abs(pontox1 - x)
 cy = abs(pontox2 - y)

 x = pontox1
 y = pontox2
 it=it+1
 iteracao = np.append(iteracao,it)

plt.plot(iteracao,img)                       #curva de convergencia

curvas_de_nivel() 
grafico_3d()


plt.show()







   