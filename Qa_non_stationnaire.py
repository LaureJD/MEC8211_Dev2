## MEC8211 - Devoir 2 - Question a)
## Auteurs : Laure Jalbert-Drouin
## Creation : 27/02/2026
# -*- coding: utf-8 -*-

"""
MEC8211 - Devoir 2 - Question A
Résolution non stationnaire en r (cylindrique) de :
    0 = Deff * [ (1/r) d/dr ( r dC/dr ) ] - S
BC: C(R)=Ce (Dirichlet), dC/dr|_{r=0}=0 (symétrie)
Discrétisation: différences finies 2e ordre (maillage uniforme)

Sorties :
- Figure 1 : profil C(r) numérique vs analytique

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# Paramètres physiques (modifiez au besoin)
# ----------------------------
DEFF = 1.0e-10  # m^2/s
k = 4.0e-9  # s^-1 
CE = 20.0  # mol/m^3
R = 0.5  # m  (rayon, D=1 m)

# ----------------------------
# Fonctions utilitaires
# ----------------------------

def resout_transitoire_radial(n, deff, k, ce, R, dt, t_final, c_init=None):
    """
    Résout le problème non stationnaire en r sur N points (i=0..N-1) avec r_N=R,
    schéma implicite Backward Euler en temps + DF2 en espace.

    PDE:
        (C^{n+1}_i - C^n_i)/dt = deff * [ C'' + (1/r) C' ]^{n+1}_i - k * C^{n+1}_i

    Discrétisation 2e ordre :
        - centre (i=0) : C''(0) ≈ 2*(C1 - C0)/dr^2 (équiv. à dC/dr=0, symétrie)
        - intérieurs (i=1..N-2) : (1/r) d/dr(r dC/dr) ≈ C'' + (1/r) C'
        - bord (i=N-1) : Dirichlet C=Ce

    Paramètres
    ---------
    n : nombre de noeuds radiaux (incluant 0 et R)
    deff : diffusivité effective (m^2/s)
    k : constante de réaction (s^-1)
    ce : concentration imposée au bord (mol/m^3)
    R : rayon à l'extrémité(m)
    dt : pas de temps (s)
    t_final : temps final de simulation (s)
    C0 : condition initiale nulle : C(r,0)=0.

    Renvoie
    -------
    r : ndarray (N,)
        coordonnées radiales
    t : ndarray (Nt+1,)
        instants de temps
    C : ndarray (Nt+1, N)
        solution : C[n_time, i_r]
    """

    # Maillage spatial
    n = int(n)
    dr = R / (n - 1)
    r = np.linspace(0.0, R, n)

    # Maillage temporel
    Nt = int(np.round(t_final / dt))
    t = np.linspace(0.0, Nt * dt, Nt + 1)

    # Condition initiale
    Cn = np.zeros(n, dtype=float)

    # Tableau solution
    C = np.zeros((Nt + 1, n), dtype=float)
    C[0, :] = Cn

    # Construction de la matrice A (constante dans le temps) et du vecteur b
    a_mat = np.zeros((n, n), dtype=float)

    a_mat[0, 0] = (1.0 / dt) + k + 2.0 * deff / dr**2   # i=0 : centre (Gear)
    a_mat[0, 1] = -2.0 * deff / dr**2
    
    for i in range(1, n - 1):           # i = 1..N-2 (intérieurs)
        ri = r[i]
        aW = -deff / dr**2 + deff / (2.0 * ri * dr)
        aP = (1.0 / dt) + k + 2.0 * deff / dr**2
        aE = -deff / dr**2 - deff / (2.0 * ri * dr)
        a_mat[i, i - 1] = aW
        a_mat[i, i] = aP
        a_mat[i, i + 1] = aE

    a_mat[n - 1, n - 1] = 1.0           # i = N-1 (bord) : Dirichlet

    invA = np.linalg.inv(a_mat)         # inverse de A 

    # Construction du vecteur b 
    b_vec = np.zeros(n, dtype=float)

    #Cacule de C^{n+1} à chaque pas de temps
    for nstep in range(Nt):
        b_vec[:] = Cn / dt           # Cn[0..n-2]/dt, la dernière valeur sera écrasée
        b_vec[-1] = ce               # condition au bord

        Cnp1 = invA @ b_vec             # multiplication par l'inverse de A pour obtenir C^{n+1

        C[nstep + 1, :] = Cnp1
        Cn = Cnp1

    return r, t, C



# ----------------------------
# a) Profils de concentration
# ----------------------------
if __name__ == "__main__":
    N = 11  # nombre de points radiaux (incluant 0 et R)

    r, t, c= resout_transitoire_radial(N, DEFF, k, CE, R, dt=1000, t_final=4.0e9)

    # --- Figure 1 : profil 3D C(r,t)

    # Rmesh, Tmesh = np.meshgrid(r, t)  # maillages 2D

    # fig = plt.figure(figsize=(7, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(Rmesh, Tmesh, c, cmap='viridis', linewidth=0, antialiased=True)
    # ax.set_xlabel('r (m)')
    # ax.set_ylabel('t (s)')
    # ax.set_zlabel('C (mol/m³)')
    # ax.set_title('Surface 3D : C(r,t)')
    # fig.colorbar(surf, shrink=0.7, aspect=20, label='C (mol/m³)')
    # plt.tight_layout()
    # plt.show()


 # --- Figure 2 : courbes C(t) pour plusieurs rayons
    r_values = [0.0, 0.25*R, 0.5*R, 0.75*R, 0.90*R] 
    plt.figure(figsize=(6.6, 4.6))
    
    # for rv in r:
    #     i = np.argmin(np.abs(r - rv))
    #     label = fr"r = {r[i]:.3f} m"
    #     plt.plot(t, c[:, i], lw=2, label=label)

    # plt.xlabel('t (s)')
    # plt.ylabel('C (mol/m³)')
    # plt.title('Séries temporelles C(t) à plusieurs rayons pour N=11')
    # plt.grid(True, alpha=0.3)
    # plt.legend(loc='upper right')
    # plt.show()

    plt.figure(figsize=(6.6, 4.6))
    #for tv in [0, 1000, 2000, 1e3, 1e6, 1e9 ] :
    for tv in [1e6, 1e7, 1e8, 1e9, 4e9 ] :
        j = np.argmin(np.abs(t - tv))
        label = fr"t = {t[j]:.2e} s"
        plt.plot(r, c[j, :], lw=2, label=label)
    plt.xlabel('r (m)')
    plt.ylabel('C (mol/m³)')
    plt.title('Profils radiaux C(r) à plusieurs temps pour N=11')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.show()