# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 00:26:25 2025

@author: User


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm, lognorm, chi2, pareto, zipf, uniform, randint

# ==========================================
# QUESTION 1 : FONCTIONS DE VISUALISATION
# ==========================================

def tracer_discretes():
    plt.figure(figsize=(15, 10))
    
    # 1. Dirac (centrée en 5)
    plt.subplot(2, 3, 1)
    x_dirac = np.arange(0, 11)
    y_dirac = np.where(x_dirac == 5, 1, 0)
    plt.stem(x_dirac, y_dirac)
    plt.title("Loi de Dirac (x=5)")

    # 2. Uniforme Discrète (Dé à 6 faces)
    plt.subplot(2, 3, 2)
    x_unif = np.arange(1, 7)
    plt.bar(x_unif, randint.pmf(x_unif, 1, 7), alpha=0.7)
    plt.title("Uniforme Discrète (1 à 6)")

    # 3. Binomiale
    plt.subplot(2, 3, 3)
    n, p = 10, 0.5
    x_binom = np.arange(0, n+1)
    plt.bar(x_binom, binom.pmf(x_binom, n, p), color='g', alpha=0.7)
    plt.title(f"Binomiale (n={n}, p={p})")

    # 4. Poisson
    plt.subplot(2, 3, 4)
    mu = 4
    x_pois = np.arange(0, 15)
    plt.bar(x_pois, poisson.pmf(x_pois, mu), color='r', alpha=0.7)
    plt.title(f"Poisson (lambda={mu})")

    # 5. Zipf-Mandelbrot
    plt.subplot(2, 3, 5)
    a = 2.0 
    x_zipf = np.arange(1, 11)
    plt.bar(x_zipf, zipf.pmf(x_zipf, a), color='purple', alpha=0.7)
    plt.title(f"Zipf (a={a})")

    plt.suptitle("DISTRIBUTIONS DISCRÈTES")
    plt.tight_layout()
    plt.show()

def tracer_continues():
    plt.figure(figsize=(15, 10))
    x = np.linspace(0.1, 10, 100)

    # 1. Normale
    plt.subplot(2, 3, 1)
    plt.plot(x, norm.pdf(x, 5, 1))
    plt.fill_between(x, norm.pdf(x, 5, 1), alpha=0.2)
    plt.title("Normale (mu=5, sigma=1)")

    # 2. Log-Normale
    plt.subplot(2, 3, 2)
    plt.plot(x, lognorm.pdf(x, 0.95))
    plt.title("Log-Normale")

    # 3. Uniforme Continue
    plt.subplot(2, 3, 3)
    x_u = np.linspace(0, 6, 100)
    plt.plot(x_u, uniform.pdf(x_u, 1, 4)) # de 1 à 5
    plt.title("Uniforme Continue (1 à 5)")

    # 4. Chi-deux
    plt.subplot(2, 3, 4)
    plt.plot(x, chi2.pdf(x, df=3), color='orange')
    plt.title("Chi-deux (k=3)")

    # 5. Pareto
    plt.subplot(2, 3, 5)
    plt.plot(x, pareto.pdf(x, b=2.6), color='red')
    plt.title("Pareto (b=2.6)")

    plt.suptitle("DISTRIBUTIONS CONTINUES")
    plt.tight_layout()
    plt.show()

# ==========================================
# QUESTION 2 : CALCULS DES STATISTIQUES
# ==========================================

def calculer_statistiques():
    print("\n" + "="*50)
    print(f"{'DISTRIBUTION':<25} | {'MOYENNE':<10} | {'ÉCART-TYPE':<10}")
    print("-" * 50)

    # Liste des configurations (Nom, objet scipy, paramètres)
    config = [
        ("Binomiale (n=10, p=0.5)", binom, {'n': 10, 'p': 0.5}),
        ("Poisson (lambda=4)", poisson, {'mu': 4}),
        ("Zipf (a=2.0)", zipf, {'a': 2.0}),
        ("Normale (mu=5, sigma=1)", norm, {'loc': 5, 'scale': 1}),
        ("Uniforme (1 à 5)", uniform, {'loc': 1, 'scale': 4}),
        ("Chi-deux (df=3)", chi2, {'df': 3}),
        ("Pareto (b=2.6)", pareto, {'b': 2.6}),
        ("Log-Normale", lognorm, {'s': 0.95})
    ]

    for nom, dist, params in config:
        # moments='mv' récupère Moyenne (m) et Variance (v)
        m, v = dist.stats(**params, moments='mv')
        sigma = np.sqrt(v)
        print(f"{nom:<25} | {float(m):<10.2f} | {float(sigma):<10.2f}")
    
    # Cas spécial : Dirac (n'existe pas directement dans scipy.stats)
    print(f"{'Dirac (x=5)':<25} | {5.00:<10.2f} | {0.00:<10.2f}")
    print("="*50)

if __name__ == "__main__":
    # 1. On affiche les calculs dans la console
    calculer_statistiques()
    
    # 2. On affiche les graphiques
    tracer_discretes()
    tracer_continues()    





