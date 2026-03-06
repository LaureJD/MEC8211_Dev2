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

# ----------------------------
# Solveur instationnaire (BE + DF2 + Gear)
# ----------------------------
def resout_mms(n, deff, k, r_max, dt, t_final):
    """
    Résout:
        C_t = Deff*(C'' + (1/r)C') - k*C + f_MMS
    CI: C(r,0) = C_MMS(r,0) = 0
    CL: centre -> Gear ; bord -> Dirichlet g(t)=0
    """
    n = int(n)
    dr = r_max / (n - 1)
    r = np.linspace(0.0, r_max, n)

    Nt = int(np.round(t_final / dt))
    t = np.linspace(0.0, Nt*dt, Nt+1)

    # CI = C_MMS(r,0) = 0  (car sin(0)=0)
    Cn = np.zeros(n, dtype=float)
    C  = np.zeros((Nt+1, n), dtype=float)
    C[0, :] = Cn

    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for nstep in range(Nt):
        tn1 = t[nstep+1]
        A.fill(0.0); b.fill(0.0)

        # i=0 : Gear (Neumann)  ->  (-3 C0 + 4 C1 - C2) / (2 dr) = 0
        A[0,0] = -3.0
        A[0,1] =  4.0
        A[0,2] = -1.0
        b[0]   =  0.0

        # i=1..N-2 : BE + DF2
        for i in range(1, n-1):
            ri = r[i]
            aW = -deff / dr**2 + deff / (2.0 * ri * dr)
            aP =  (1.0 / dt) + k + 2.0 * deff / dr**2
            aE = -deff / dr**2 - deff / (2.0 * ri * dr)

            A[i,i-1] = aW
            A[i,i]   = aP
            A[i,i+1] = aE

            # RHS: C^n/dt + f^{n+1}
            b[i] = Cn[i]/dt + f_mms(ri, tn1, deff, k, r_max)

        # i=N-1 : Dirichlet g(t^{n+1}) = 0
        A[n-1,:]   = 0.0
        A[n-1,n-1] = 1.0
        b[n-1]     = 0.0

        Cnp1 = np.linalg.solve(A, b)
        C[nstep+1,:] = Cnp1
        Cn = Cnp1

    return r, t, C


# ----------------------------
# MAIN — Graphiques
# ----------------------------
if __name__ == "__main__":

    
    N       = 101
    DT      = 5e-3
    T_FINAL = 2.0
    r_mms, t_mms, C_mms = resout_mms(N, DEFF, K, R, DT, T_FINAL)
    # print(r_mms)

    r_values = [0.0, 0.25*R, 0.5*R, 0.75*R, 0.9*R] 
    plt.figure(figsize=(6.6, 4.6))
    for rv in r_values:
        i = np.argmin(np.abs(r - rv))
        print(i)
        label = fr"r = {i:.3f} m"
        plt.plot(t_mms, C_mms[:, i], lw=2, label=label)

    plt.xlabel('t (s)')
    plt.ylabel('C (mol/m³)')
    plt.title('Séries temporelles C(t) à plusieurs rayons')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

