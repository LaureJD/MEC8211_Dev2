## MEC8211 - Devoir 2 - Question b) (MMS simplifiée, version symbolique)
## Auteurs : Laure Jalbert-Drouin
## Creation : 27/02/2026
# -*- coding: utf-8 -*-

"""
MMS choisie (meilleur compromis) :
    C_MMS(r,t) = (1 - (r/R)^2) * sin(t)

PDE:
    C_t = Deff*(C_rr + (1/r) C_r) - k*C + f(r,t)

BCs :
    - centre r=0 : Neumann (symétrie)  ∂C/∂r = 0  (schéma de Gear)
    - bord  r=R : Dirichlet g(t) = C_MMS(R,t) = 0

Démarche :
    - génération symbolique (SymPy) de f(r,t)
    - passage en NumPy (lambdify) pour l'usage dans le solveur
    - résolution instationnaire (BE + DF2 + Gear)
    - tracés : C_MMS, f_MMS
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import Qa_non_stationnaire as Qa

# ----------------------------
# Paramètres physiques pour MMS
# ----------------------------
DEFF = 1.0e-2   # m^2/s (choix pour MMS : simple et rapide)
K    = 1.0e-1   # s^-1
R    = 0.5      # m

# ----------------------------
# Définition symbolique de la MMS
# ----------------------------
r, t, R_sym = sp.symbols('r t R', positive=True, real=True)
Deff, k = sp.symbols('Deff k', positive=True, real=True)

C_sym = (1 - (r/R_sym)**2) * sp.sin(t)  # MMS
Ct_sym  = sp.diff(C_sym, t)
Cr_sym  = sp.diff(C_sym, r)
Crr_sym = sp.diff(Cr_sym, r)

L_sym = Crr_sym + (1/r)*Cr_sym
f_sym = sp.simplify(Ct_sym - Deff*L_sym + k*C_sym)
g_sym = sp.simplify(C_sym.subs(r, R_sym))  # ici g(t)=0

# Passage en NumPy
C_mms = sp.lambdify((r, t, R_sym), C_sym, 'numpy')
f_mms = sp.lambdify((r, t, Deff, k, R_sym), f_sym, 'numpy')
g_dir = sp.lambdify((t, R_sym), g_sym, 'numpy')

print(C_mms)
print(f_mms)
print(g_dir)
# ----------------------------
# MAIN — Graphiques
# ----------------------------
if __name__ == "__main__":

    # Paramètres de résolution
    N       = 101
    DT      = 5e-3
    T_FINAL = 20.0
    # Maillage spatial
    n = int(N)
    dr = R / (n - 1)
    r = np.linspace(0.0, R, n)

    # Maillage temporel
    Nt = int(np.round(T_FINAL / DT))
    t = np.linspace(0.0, Nt * DT, Nt + 1)
    
    #Tracer MMS
    C = C_mms(r[:, None], t[None, :], R)
    r_values = [0.0, 0.25*R, 0.5*R, 0.75*R, 0.9*R] 
    plt.figure(figsize=(6.6, 4.6))
    for rv in r_values:
        i = np.argmin(np.abs(r - rv))
        label = fr"r = {r[i]:.3f} m"
        plt.plot(t, C[i, :], lw=2, label=label)

    plt.xlabel('t (s)')
    plt.ylabel('C_MMS (mol/m³)')
    plt.title('Séries temporelles C_MMS(t) à plusieurs rayons')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.show()

    #Tracer terme source f_MMS
    F = f_mms(r[:, None], t[None, :], DEFF, K, R)
    
    r_values = [0.0, 0.25*R, 0.5*R, 0.75*R, 0.9*R] 
    plt.figure(figsize=(6.6, 4.6))
    for rv in r_values:
        i = np.argmin(np.abs(r - rv))
        label = fr"r = {r[i]:.3f} m"
        plt.plot(t, F[i, :], lw=2, label=label)

    plt.xlabel('t (s)')
    plt.ylabel('f_MMS (mol/m³/s)')
    plt.title('Séries temporelles f_MMS(t) à plusieurs rayons')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

