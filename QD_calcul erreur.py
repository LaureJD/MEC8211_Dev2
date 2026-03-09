# -*- coding: utf-8 -*-
"""
MEC8211 - Devoir 2 - Solution Complète Autonome (MMS + Solveur)
Auteur : Laure Jalbert-Drouin (Adapté)
Date : 27/02/2026
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =============================================================================
# 1. GÉNÉRATION DE LA SOLUTION MANUFACTURÉE (MMS) - SYMPY
# =============================================================================
def generer_mms():
    r_s, t_s, R_s = sp.symbols('r t R', positive=True, real=True)
    Deff_s, k_s = sp.symbols('Deff k', positive=True, real=True)

    # MMS choisie : C = (1 - (r/R)^2) * sin(t)
    C_sym = (1 - (r_s/R_s)**2) * sp.sin(t_s)

    # Calcul des dérivées pour l'EDP : Ct = Deff*(Crr + 1/r*Cr) - k*C + f
    Ct_s = sp.diff(C_sym, t_s)
    Cr_s = sp.diff(C_sym, r_s)
    Crr_s = sp.diff(Cr_s, r_s)
    
    # Terme source f(r,t)
    f_sym = sp.simplify(Ct_s - Deff_s * (Crr_s + (1/r_s)*Cr_s) + k_s * C_sym)

    # Conversion en fonctions utilisables par NumPy
    c_func = sp.lambdify((r_s, t_s, R_s), C_sym, 'numpy')
    f_func = sp.lambdify((r_s, t_s, Deff_s, k_s, R_s), f_sym, 'numpy')
    
    return c_func, f_func

# =============================================================================
# 2. SOLVEUR NUMÉRIQUE (DIFFÉRENCES FINIES)
# =============================================================================
def solveur_numérique(n, deff, k, ce, r_max, dt, t_final, source_func):
    # Maillage spatial
    dr = r_max / (n - 1)
    r = np.linspace(0.0, r_max, n)

    # Maillage temporel
    Nt = int(np.round(t_final / dt))
    t = np.linspace(0.0, Nt * dt, Nt + 1)

    # Initialisation (C=0 à t=0 car sin(0)=0 dans notre MMS)
    Cn = np.zeros(n)
    C_resultat = np.zeros((Nt + 1, n))
    C_resultat[0, :] = Cn

    # Construction de la matrice A (Implicite)
    A = np.zeros((n, n))
    
    # i = 0 : Symétrie au centre C''(0) ≈ 2*(C1 - C0)/dr^2
    A[0, 0] = (1.0/dt) + k + (2.0 * deff / dr**2)
    A[0, 1] = -2.0 * deff / dr**2

    # i = 1 à N-2 : Points intérieurs
    for i in range(1, n - 1):
        ri = r[i]
        A[i, i-1] = -deff/dr**2 + deff/(2.0*ri*dr)
        A[i, i]   = (1.0/dt) + k + 2.0*deff/dr**2
        A[i, i+1] = -deff/dr**2 - deff/(2.0*ri*dr)

    # i = N-1 : Condition de Dirichlet au bord
    A[n-1, n-1] = 1.0

    invA = np.linalg.inv(A)

    # Boucle temporelle
    for nstep in range(Nt):
        t_nplus1 = t[nstep + 1]
        
        # Vecteur b
        b = Cn / dt
        # Ajout de la source MMS calculée par SymPy
        b += source_func(r, t_nplus1, deff, k, r_max)
        
        # Forcer Dirichlet au bord (C=0 pour cette MMS)
        b[-1] = ce 

        # Résolution
        Cnp1 = invA @ b
        C_resultat[nstep+1, :] = Cnp1
        Cn = Cnp1

    return r, t, C_resultat

# =============================================================================
# 3. EXÉCUTION ET COMPARAISON
# =============================================================================
if __name__ == "__main__":
    # Paramètres
    N_POINTS = 41
    DEFF_VAL = 0.01
    K_VAL = 0.1
    R_VAL = 0.5
    DT_VAL = 0.01
    T_END = 1.5

    # 1. Obtenir les fonctions MMS
    c_exact_func, f_source_func = generer_mms()

    # 2. Lancer le solveur
    r_coords, t_coords, C_num = solveur_numérique(
        N_POINTS, DEFF_VAL, K_VAL, 0.0, R_VAL, DT_VAL, T_END, f_source_func
    )

    # 3. Calculer la solution exacte sur la même grille
    R_mesh, T_mesh = np.meshgrid(r_coords, t_coords)
    C_exact = c_exact_func(R_mesh, T_mesh, R_VAL)

    # 4. Calcul de l'erreur
    erreur = np.abs(C_num - C_exact)
    print(f"Erreur maximale : {np.max(erreur):.2e}")

    # 5. Graphique de validation
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_coords, C_num[-1, :], 'ro', label='Numérique', markersize=4)
    plt.plot(r_coords, C_exact[-1, :], 'k-', label='Exact (MMS)')
    plt.title(f'Profil au temps t={T_END}s')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.contourf(R_mesh, T_mesh, erreur, cmap='viridis')
    plt.colorbar(label='Erreur absolue')
    plt.title('Carte de l\'erreur (r, t)')
    plt.xlabel('Rayon r')
    plt.ylabel('Temps t')

    plt.tight_layout()
    plt.show()