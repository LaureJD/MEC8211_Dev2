# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 08:39:42 2026

@author: wadih

MEC8211 - Devoir 2 - Question A
Résolution non stationnaire en r (cylindrique) avec schéma implicite

Created on: 27/02/2026
@author: Laure Jalbert-Drouin (refactored by Assistant)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Parameters():
    """Container for all physical parameters"""
    def __init__(self):
        self.DEFF = 1.0e-10  # m^2/s
        self.k = 4.0e-9      # s^-1 
        self.CE = 20.0        # mol/m^3
        self.R = 0.5          # m (rayon)
        self.N = 11           # nombre de points radiaux
        self.DT = 10000       # s (pas de temps)
        self.T_FINAL = 4.0e9  # s
        
        self.BASH_SCRIPT = True # for automation 

class TransientRadialSolver():
    """
    Solveur pour problème non stationnaire en coordonnées cylindriques:
        ∂C/∂t = Deff * [ (1/r) ∂/∂r ( r ∂C/∂r ) ] - k*C
    BC: C(R)=Ce (Dirichlet), dC/dr|_{r=0}=0 (symétrie)
    """
    
    def __init__(self, n_points, deff, k, ce, radius, dt, t_final, c_init=None):
        """
        Initialise le solveur avec les paramètres du problème
        
        Parameters
        ----------
        n_points : int
            nombre de noeuds radiaux
        deff : float
            diffusivité effective (m^2/s)
        k : float
            constante de réaction (s^-1)
        ce : float
            concentration au bord (mol/m³)
        radius : float
            rayon R (m)
        dt : float
            pas de temps (s)
        t_final : float
            temps final (s)
        c_init : float or array, optional
            condition initiale
        """
        self.n_points = int(n_points)
        self.deff = deff
        self.k = k
        self.ce = ce
        self.radius = radius
        self.dt = dt
        self.t_final = t_final
        self.c_init = c_init
        
        # Maillages
        self.r, self.dr = self._create_spatial_mesh()
        self.t, self.n_time_steps = self._create_temporal_mesh()
        
        # Matrice du système et son inverse
        self.A_matrix = self._build_matrix()
        self.invA = np.linalg.inv(self.A_matrix)
        
        # Solution
        self.solution = None
        
    def _create_spatial_mesh(self):
        """Crée le maillage spatial"""
        dr = self.radius / (self.n_points - 1)
        r = np.linspace(0.0, self.radius, self.n_points)
        return r, dr
    
    def _create_temporal_mesh(self):
        """Crée le maillage temporel"""
        n_time_steps = int(np.round(self.t_final / self.dt))
        if n_time_steps > 200000:
            print(f"Warning: number of time steps Nt={n_time_steps} is very large")
        t = np.linspace(0.0, n_time_steps * self.dt, n_time_steps + 1)
        return t, n_time_steps
    
    def _build_matrix(self):
        """
        Construit la matrice A pour le schéma implicite
        A * C^{n+1} = C^n/dt + termes de bord
        """
        A = np.zeros((self.n_points, self.n_points), dtype=float)
        
        # i = 0 (centre) : condition de symétrie
        A[0, 0] = (1.0 / self.dt) + self.k + 2.0 * self.deff / self.dr**2
        A[0, 1] = -2.0 * self.deff / self.dr**2
        
        # i = 1..N-2 (noeuds intérieurs)
        for i in range(1, self.n_points - 1):
            ri = self.r[i]
            aW = -self.deff / self.dr**2 + self.deff / (2.0 * ri * self.dr)
            aP = (1.0 / self.dt) + self.k + 2.0 * self.deff / self.dr**2
            aE = -self.deff / self.dr**2 - self.deff / (2.0 * ri * self.dr)
            
            A[i, i - 1] = aW
            A[i, i] = aP
            A[i, i + 1] = aE
        
        # i = N-1 (bord) : condition de Dirichlet
        A[self.n_points - 1, self.n_points - 1] = 1.0
        
        return A
    
    def _initialize_solution(self):
        """Initialise le tableau de solution"""
        if self.c_init is None:
            Cn = np.zeros(self.n_points, dtype=float)
        else:
            c0 = np.array(self.c_init, dtype=float)
            Cn = c0 if c0.size == self.n_points else np.full(self.n_points, float(self.c_init), dtype=float)
        
        C = np.zeros((self.n_time_steps + 1, self.n_points), dtype=float)
        C[0, :] = Cn
        return C, Cn
    
    def solve(self):
        """
        Résout le problème transitoire
        
        Returns
        -------
        tuple : (r, t, solution)
        """
        C, Cn = self._initialize_solution()
        b_vec = np.zeros(self.n_points, dtype=float)
        
        # Boucle en temps
        for nstep in range(self.n_time_steps):
            # Construction du vecteur b
            b_vec[:] = Cn / self.dt
            b_vec[-1] = self.ce  # condition au bord
            
            # Résolution
            Cnp1 = self.invA @ b_vec
            
            # Stockage
            C[nstep + 1, :] = Cnp1
            Cn = Cnp1
        
        self.solution = C
        return self.r, self.t, C
    
    def extract_radial_profiles(self, times):
        """
        Extrait les profils radiaux à des instants spécifiques
        
        Parameters
        ----------
        times : list
            Liste des instants désirés
        
        Returns
        -------
        dict : {temps: profil radial}
        """
        if self.solution is None:
            raise ValueError("La solution n'a pas encore été calculée. Appelez solve() d'abord.")
        
        profiles = {}
        for t_desired in times:
            idx = np.argmin(np.abs(self.t - t_desired))
            profiles[t_desired] = self.solution[idx, :]
        
        return profiles
    
    def extract_time_series(self, radii):
        """
        Extrait les séries temporelles à des rayons spécifiques
        
        Parameters
        ----------
        radii : list
            Liste des rayons désirés
        
        Returns
        -------
        dict : {rayon: série temporelle}
        """
        if self.solution is None:
            raise ValueError("La solution n'a pas encore été calculée. Appelez solve() d'abord.")
        
        series = {}
        for r_desired in radii:
            idx = np.argmin(np.abs(self.r - r_desired))
            series[r_desired] = self.solution[:, idx]
        
        return series


class Plotter():
    """Classe pour la visualisation des résultats"""
       
    @staticmethod
    def plot_time_series(r, t, c, selected_radii_fraction=None):
        """
        Trace les séries temporelles C(t) pour plusieurs rayons
        
        Parameters
        ----------
        r : array
            coordonnées radiales
        t : array
            instants
        c : array
            solution 2D
        selected_radii_fraction : list, optional
            fractions du rayon à tracer (ex: [0.0, 0.25, 0.5, 0.75, 0.9])
        """
        if selected_radii_fraction is None:
            selected_radii_fraction = [0.0, 0.25, 0.5, 0.75, 0.9]
        
        r_max = r[-1]
        r_values = [frac * r_max for frac in selected_radii_fraction]
        
        plt.figure(figsize=(6.6, 4.6))
        for rv in r_values:
            i = np.argmin(np.abs(r - rv))
            label = fr"r = {r[i]:.3f} m"
            plt.plot(t, c[:, i], lw=2, label=label)
        
        plt.xlabel('t (s)')
        plt.ylabel('C (mol/m³)')
        plt.title('Séries temporelles C(t) à plusieurs rayons')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_radial_profiles(r, profiles_dict):
        """
        Trace les profils radiaux à différents instants
        
        Parameters
        ----------
        r : array
            coordonnées radiales
        profiles_dict : dict
            {temps: profil radial}
        """
        plt.figure(figsize=(6.6, 4.6))
        for t_val, profile in profiles_dict.items():
            plt.plot(r, profile, lw=2, label=f't = {t_val:.1e} s')
        
        plt.xlabel('r (m)')
        plt.ylabel('C (mol/m³)')
        plt.title('Profils radiaux C(r) à différents instants')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    """Fonction principale pour exécuter la simulation"""
    
    # Paramètres
    params = Parameters()
    
    print("=" * 60)
    print("MEC8211 - Devoir 2 - Question A")
    print("Résolution non stationnaire en coordonnées cylindriques")
    print("=" * 60)
    print(f"\nParamètres physiques:")
    print(f"  Deff = {params.DEFF:.1e} m²/s")
    print(f"  k = {params.k:.1e} s⁻¹")
    print(f"  Ce = {params.CE} mol/m³")
    print(f"  R = {params.R} m")
    print(f"\nParamètres numériques:")
    print(f"  Nombre de noeuds: {params.N}")
    print(f"  Pas de temps: {params.DT} s")
    print(f"  Temps final: {params.T_FINAL:.1e} s")
    
    # Création et résolution
    solver = TransientRadialSolver(
        n_points=params.N,
        deff=params.DEFF,
        k=params.k,
        ce=params.CE,
        radius=params.R,
        dt=params.DT,
        t_final=params.T_FINAL
    )
    
    print("\nRésolution en cours...")
    r, t, c = solver.solve()
    print("Résolution terminée!")
    
    if params.BASH_SCRIPT == False:
        # Visualisation
        plotter = Plotter()
        
        # Série temporelle à différents rayons
        plotter.plot_time_series(r, t, c)
        
        # Profils radiaux à différents instants
        times_of_interest = [1e7, 1e8, 1e9, 4e9]
        profiles = solver.extract_radial_profiles(times_of_interest)
        plotter.plot_radial_profiles(r, profiles)
        
    
    return solver


if __name__ == "__main__":
    solver = main()