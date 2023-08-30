import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize


def function(x, m, b):
    return m*x + b

def get_data(nombre_archivo):
    datos = pd.read_csv(nombre_archivo, delimiter=";", decimal=",")
    keys = datos.keys()

    f = np.array(datos[keys[0]])*1E6
    B = np.array(datos[keys[1]])*1E-3

    popt, pcov = scipy.optimize.curve_fit(function, B, f)
    return B, f, popt, pcov

def generador_etiquetas(datos, nombres):
    etiquetas = []

    for i in range(len(datos)):    
        etiquetas.append(nombres[i])

        popt = datos[i][2]
        pcov = datos[i][3]

        sigmas = np.sqrt(np.diag(pcov))

        m = popt[0]/1E8
        sigma_m = sigmas[0]/1E8

        b = popt[1]/1E6
        sigma_b = sigmas[1]/1E6

        ajuste = "$f = mB + b$\n"
        ajuste += "$m$ = ({:.2f} $\pm$ {:.2f}) x 10$^8$ Hz/T\n".format(m, sigma_m)
        ajuste += "$b$ = ({:.1f} $\pm$ {:.1f}) x 10$^6$ Hz\n".format(b, sigma_b)

        etiquetas.append(ajuste)

    return etiquetas

def grafica(datos, colores, etiquetas):
    plt.figure()

    for i in range(len(datos)):
        B, f, popt, pcov = datos[i]
        
        material = etiquetas[2*i]
        ajuste = etiquetas[2*i + 1]

        plt.scatter(B/1E-3, f/1E6, c=colores[i], label=material, s=25)
        plt.plot(B/1E-3, function(B, *popt)/1E6, c=colores[i], label=ajuste)

        plt.errorbar(B/1E-3, f/1E6, xerr=0.1, yerr=0.0001, fmt=".", c=colores[i])
    
    plt.title("Frecuencia ($f$) vs Campo magnético ($B$)")
    plt.legend()
    plt.xlabel("$B$ (mT)")
    plt.ylabel("$f$ (MHz)")
    plt.savefig("Grafica-fvsB.png")
    return None


data_teflon = get_data("f-B-teflon.csv")
data_glicerina = get_data("f-B-glicerina.csv")

datos = [data_teflon, data_glicerina]
colores = ["black", "red"]
nombres = ["Teflón", "Glicerina"]

etiquetas = generador_etiquetas(datos, nombres)

grafica(datos, colores, etiquetas)

#Calculos

h_bar = 1.054571817E-34
mu_N = 5.050783E-27

m = 1.14*1E8
sigma_1 = 0.04*1E8
sigma_2 = 0.02*1E8

gamma_teflon = 2*np.pi*m
gamma_glicerina = 2*np.pi*m

sigma1_gamma = 2*np.pi*sigma_1
sigma2_gamma = 2*np.pi*sigma_2

g_teflon = gamma_teflon*h_bar/mu_N
g_glicerina = gamma_glicerina*h_bar/mu_N

sigma1_g = sigma1_gamma*h_bar/mu_N
sigma2_g = sigma2_gamma*h_bar/mu_N

print(g_teflon)
print(sigma1_g)

print(g_glicerina)
print(sigma2_g)





plt.show()