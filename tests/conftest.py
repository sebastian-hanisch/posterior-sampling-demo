"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 7 (symmetrisch, Zeilensumme 1, Singulärwerte 1, 0,5, 0,1), Daten b = (1,3, 1,4, 1,3) (rauschfrei für x = (1, 2, 1)), Rauschen sigma = 0,1,
    Gaußscher Prior N(0, tau^2 I) mit tau = 1: der exponentielle Kern auf drei weit entfernten Punkten ist bis auf 1e-36 die Einheitsmatrix (`s = (0, 10, 20)`).
    Handrechnung (Stück 7): Posterior-Mittel (1,153465, 1,653465, 1,153465), Posterior-Eigenwerte 1/101, 1/26, 1/2 (Gamma_11 = 0,105864, Gamma_22 = 0,336634, Gamma_13 = 0,067403).
    Likelihood-Potential Phi(x) = |A x - b|^2 / (2 sigma^2): Phi(1, 2, 1) = 0, Phi(0) = 5,34 / 0,02 = 267. Prior-Terme für x = (1, 2, 1): Gauß 1/2 |x|^2 = 3, Laplace (Skala 0,5)
    |x|_1 / 0,5 = 8, pseudo-Huber (eps = 0,1, Skala 0,5) (4,012474 - 0,3) / 0,5 = 7,424948. Identifikation: die Eigenwerte tau^2 sv^2 / sigma^2 sind für sigma = 0,2 (25, 6,25, 0,25)."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b


@pytest.fixture
def mini_target(mini):
    import smp_model as m

    a, x, b = mini
    return m.Target(a, b, 0.1, "gauss", np.array([0.0, 10.0, 20.0]), tau=1.0, ell=0.12)
