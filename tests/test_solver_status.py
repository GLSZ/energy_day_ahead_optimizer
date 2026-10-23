"""
Tests de gestion des statuts du solveur par optimizer.solve() -- demande
explicite de la Correction 4 : optimal, infaisable, non borne.

optimizer.solve() accepte n'importe quel pulp.LpProblem : on teste ici
avec des problemes LP volontairement minuscules et artificiels (pas
derives du jeu de donnees reel du projet), ce qui est la methode
standard pour tester unitairement une fonction generique autour d'un
solveur -- ce n'est pas "remplacer des donnees reelles par des donnees
fictives" (qui concerne les resultats du pipeline metier), juste
verifier le comportement de solve() face a chaque statut possible.
"""
import pulp
import pytest

import optimizer


def test_solve_returns_optimal_for_a_simple_feasible_problem():
    prob = pulp.LpProblem("trivial_optimal", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0, upBound=10)
    prob += x, "obj"
    prob += x <= 5, "cap"

    status = optimizer.solve(prob)
    assert status == "Optimal"
    assert pulp.value(x) == pytest.approx(5.0)


def test_solve_raises_on_infeasible_problem():
    prob = pulp.LpProblem("trivial_infeasible", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0)
    prob += x, "obj"
    prob += x <= 0, "upper"
    prob += x >= 1, "lower"   # contradictoire avec x <= 0

    with pytest.raises(RuntimeError, match="Infeasible"):
        optimizer.solve(prob)


def test_solve_raises_on_unbounded_problem():
    """
    Regression : avant la Correction 4, seul le statut "Infeasible"
    etait bloque -- un probleme "Unbounded" passait silencieusement et
    affichait un "Profit optimal" trompeur (verifie manuellement avant
    correction). solve() doit desormais lever pour tout statut
    different de "Optimal".
    """
    prob = pulp.LpProblem("trivial_unbounded", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0)  # pas de borne haute
    prob += x, "obj"   # maximiser x sans aucune contrainte superieure

    with pytest.raises(RuntimeError, match="Unbounded"):
        optimizer.solve(prob)


def test_solve_does_not_raise_for_degenerate_but_optimal_problem():
    """Cas limite : un probleme avec une seule solution possible
    (bornes egales) doit rester traite comme Optimal, pas comme un cas
    d'erreur."""
    prob = pulp.LpProblem("degenerate_optimal", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=3, upBound=3)
    prob += x, "obj"

    status = optimizer.solve(prob)
    assert status == "Optimal"
    assert pulp.value(x) == pytest.approx(3.0)
