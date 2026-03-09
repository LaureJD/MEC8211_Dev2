from QD_calcul_erreur import solveur_numérique
import numpy as np
import matplotlib.pyplot as plt

#Parametre pour avoir un Da de 1
Deff = 5.0e-10  # m^2/s
k = 4.0e-2
R=0.5
n=100

Ce=0
erreur=0

def f_source_mms(r, t, Deff, k, R):
    term_t = (1 - (r/R)**2) * np.cos(t)
    laplacien = (-4 / R**2) * np.sin(t)
    term_reac = k * (1 - (r/R)**2) * np.sin(t)
    
    # f = dC/dt - Deff*Laplacien + k*C
    return term_t - (Deff * laplacien) + term_reac

def MMS(r,t):
    R=0.5
    return (1-(r/R)**2)*np.sin(t)

def calcul_Da(k, R, Deff):
    
    Da = k * R**2 / Deff
    return Da
r,t,C=solveur_numérique(n, Deff, k, Ce, R, dt=0.1, t_final=2, source_func=f_source_mms)
def L1(n, t, C):
    erreur=0
    for i in range(n):
        for j in range(len(t)):
            u_a = MMS(r[i],t[j])
            erreur += np.abs(C[j,i]-u_a)
            erreur_l1 = 1/(n*len(t))*erreur
    return erreur_l1
def L2(n, t, C):
    erreur=0
    for i in range(n):
        for j in range(len(t)):
            u_a = MMS(r[i],t[j])
            erreur += (C[j,i]-u_a)**2
            erreur_l2 = np.sqrt(1/(n*len(t))*(erreur))
    return erreur_l2
def L_inf(n, t, C):
    erreur_inf=0
    for i in range(n):
        for j in range(len(t)):
            u_a = MMS(r[i],t[j])
            erreur = C[j,i]-u_a
            if erreur_inf<np.abs(erreur):
                erreur_inf=np.abs(erreur)
    return erreur_inf
    
test= [L1(n,t,C), L2(n,t,C), L_inf(n, t, C)]
print(test)
print((calcul_Da(k , R, Deff)))