"""
Tests de bout en bout pour les deux scenarios cote a cote produits par
optimizer.run_optimization() :
  - "forecast"          -> dispatch_results_<date>.csv
  - "perfect_foresight"  -> dispatch_results_<date>_perfect_foresight.csv

Invariant mathematique verifie : les deux scenarios partagent EXACTEMENT
le meme domaine realisable (memes contraintes) ; seul l'objectif differe
(prevision vs prix realise). Le scenario 'perfect_foresight' optimise
directement la fonction qui sert aussi a evaluer le P&L (le prix
realise) -> sa marge, evaluee sur le prix realise, doit donc TOUJOURS
etre >= celle du scenario 'forecast' evaluee sur ce meme prix realise
(optimalite d'un LP sur un domaine partage). Si ce test echoue, soit le
domaine realisable differe entre les deux scenarios (bug), soit
extract_results() n'evalue plus les deux sur le meme prix (regression
de la Correction 2).
"""
import datetime

import pandas as pd
import pytest

import config
import optimizer

KNOWN_DATE = datetime.date(2026, 9, 13)


@pytest.fixture(scope="module")
def ran_optimization():
    config.TARGET_DATE = KNOWN_DATE
    results_forecast = optimizer.run_optimization()
    return results_forecast


def test_both_scenario_files_are_saved(ran_optimization):
    forecast_path = f"data/processed/dispatch_results_{KNOWN_DATE}.csv"
    perfect_path = f"data/processed/dispatch_results_{KNOWN_DATE}_perfect_foresight.csv"

    df_forecast = pd.read_csv(forecast_path)
    df_perfect = pd.read_csv(perfect_path)

    assert len(df_forecast) == 96
    assert len(df_perfect) == 96


def test_perfect_foresight_margin_is_at_least_forecast_margin(ran_optimization):
    """Invariant d'optimalite LP : voir docstring du module."""
    df_forecast = pd.read_csv(f"data/processed/dispatch_results_{KNOWN_DATE}.csv")
    df_perfect = pd.read_csv(f"data/processed/dispatch_results_{KNOWN_DATE}_perfect_foresight.csv")

    margin_forecast = df_forecast["margin_eur"].sum()
    margin_perfect = df_perfect["margin_eur"].sum()

    assert margin_perfect >= margin_forecast - 1e-3, (
        f"la borne 'connaissance parfaite' ({margin_perfect:.2f} €) est "
        f"inferieure au dispatch realiste ({margin_forecast:.2f} €) "
        "-> invariant d'optimalite viole"
    )


def test_run_optimization_still_returns_forecast_scenario_for_compat(ran_optimization):
    """
    main.py::step_optimize() utilise le retour de run_optimization()
    comme avant (results['margin_eur'].sum(), etc.) -> l'interface ne
    doit pas changer : on doit recevoir le scenario 'forecast'.
    """
    df_forecast = pd.read_csv(f"data/processed/dispatch_results_{KNOWN_DATE}.csv")
    assert ran_optimization["margin_eur"].sum() == pytest.approx(
        df_forecast["margin_eur"].sum()
    )
