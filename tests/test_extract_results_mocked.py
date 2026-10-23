"""
Teste extract_results() avec un MOCK complet de PuLP -- demande
explicite de la Correction 4.

pulp.value(x) retourne x tel quel si x est deja un nombre (voir
pulp/utilities.py: `if isNumber(x): return x`). On peut donc passer des
dicts de simples floats a la place de vraies LpVariable : aucun solveur
n'est jamais appele, le test est rapide et deterministe, et on verifie
des valeurs attendues CALCULEES A LA MAIN, pas juste l'absence de crash.

C'est exactement le test qui aurait detecte en quelques millisecondes
le bug de signature de extract_results() decouvert en Phase 1 (3
parametres declares vs 7 passes), sans avoir besoin de resoudre un LP
complet au prealable.
"""
import datetime

import pandas as pd
import pytest

import optimizer

N = 3  # 3 slots suffisent a verifier toute la logique de calcul


@pytest.fixture(autouse=True)
def small_slots(monkeypatch):
    """
    extract_results() itere sur optimizer.SLOTS (global du module,
    range(96) par defaut), pas sur len(df) -- on le reduit a range(N)
    pour ce fichier de test, revert automatique par monkeypatch apres
    chaque test (n'affecte aucun autre fichier de test).
    """
    monkeypatch.setattr(optimizer, "SLOTS", list(range(N)))
    monkeypatch.setattr(optimizer, "N_SLOTS", N)


@pytest.fixture
def mocked_df():
    return pd.DataFrame({
        "datetime": pd.date_range("2026-09-13", periods=N, freq="15min", tz="Europe/Paris"),
        "da_price_eur_mwh": [50.0, 150.0, 20.0],              # prix REALISE (evaluation)
        "da_price_forecast_eur_mwh": [55.0, 140.0, 25.0],      # prevision (non relu ici)
        "nuclear_marginal_cost": [12.0] * N,
        "gas_marginal_cost": [93.0] * N,
        "hydro_marginal_cost": [5.0] * N,
        "battery_marginal_cost": [2.0] * N,
        "demand_mw": [1000.0, 1000.0, 1000.0],
    })


@pytest.fixture
def mocked_variables():
    """Toutes les 'variables PuLP' sont ici de simples floats : voir
    docstring du module pour pourquoi pulp.value() les accepte tel quel."""
    P = {
        "nuclear": {0: 100.0, 1: 100.0, 2: 100.0},
        "gas":     {0: 0.0,   1: 200.0, 2: 0.0},
        "hydro":   {0: 10.0,  1: 10.0,  2: 10.0},
        "wind":    {0: 5.0,   1: 5.0,   2: 5.0},
        "solar":   {0: 0.0,   1: 0.0,   2: 0.0},
        "battery": {0: -20.0, 1: 30.0,  2: 0.0},   # charge, decharge, neutre
    }
    SOC     = {0: 50.0, 1: 40.0, 2: 60.0}
    V_res   = {0: 30.0, 1: 29.5, 2: 29.0}
    V_aval  = {0: 3.0,  1: 3.1,  2: 3.2}
    P_pump  = {0: 0.0,  1: 0.0,  2: 0.0}
    Q_spill = {0: 0.05, 1: 0.05, 2: 0.05}
    U_gas     = {0: 0.0, 1: 1.0, 2: 0.0}
    Start_gas = {0: 0.0, 1: 1.0, 2: 0.0}   # un seul demarrage, au slot 1
    return P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas


def test_extract_results_returns_expected_shape(mocked_df, mocked_variables):
    results_df = optimizer.extract_results(mocked_df, *mocked_variables)
    assert len(results_df) == N
    for col in ["slot", "da_price", "da_price_forecast", "revenue_eur",
                "fuel_cost_eur", "margin_eur", "u_gas", "start_gas",
                "gas_startup_cost_eur"]:
        assert col in results_df.columns


def test_extract_results_computes_expected_margin_slot_by_slot(mocked_df, mocked_variables):
    """
    Valeurs calculees a la main (voir commentaires) a partir des memes
    formules que extract_results() : revenue = prix_realise x energie,
    fuel_cost = cout_marginal x |energie| (+ cout de demarrage gaz),
    margin = revenue - fuel_cost - cout_demarrage.
    """
    results_df = optimizer.extract_results(mocked_df, *mocked_variables)

    gas_startup_cost = optimizer.ASSETS[optimizer.ASSET_CONFIG_KEY["gas"]]["startup_cost"]
    assert gas_startup_cost == 8000  # valeur reelle de config.py, pas inventee

    # Slot 0 : prix=50, nuclear 100MW+hydro 10MW+wind 5MW produisent,
    # gas=0, batterie charge 20MW (cout = prix x energie chargee).
    expected_revenue_0 = 50 * (100 + 10 + 5) * 0.25
    expected_fuel_0 = (12 * 100 + 5 * 10 + 0) * 0.25 + 50 * (20 * 0.25)
    expected_margin_0 = expected_revenue_0 - expected_fuel_0
    assert results_df.loc[0, "revenue_eur"] == pytest.approx(expected_revenue_0, abs=0.01)
    assert results_df.loc[0, "fuel_cost_eur"] == pytest.approx(expected_fuel_0, abs=0.01)
    assert results_df.loc[0, "margin_eur"] == pytest.approx(expected_margin_0, abs=0.01)
    assert results_df.loc[0, "gas_startup_cost_eur"] == 0.0

    # Slot 1 : prix=150, gas demarre (200MW, cout demarrage inclus),
    # batterie decharge 30MW (vendue au prix du marche).
    expected_revenue_1 = 150 * (100 + 200 + 10 + 5 + 30) * 0.25
    expected_fuel_1 = (12 * 100 + 93 * 200 + 5 * 10 + 2 * 30) * 0.25
    expected_startup_1 = gas_startup_cost * 1.0
    expected_margin_1 = expected_revenue_1 - expected_fuel_1 - expected_startup_1
    assert results_df.loc[1, "revenue_eur"] == pytest.approx(expected_revenue_1, abs=0.01)
    assert results_df.loc[1, "gas_startup_cost_eur"] == pytest.approx(expected_startup_1)
    assert results_df.loc[1, "margin_eur"] == pytest.approx(expected_margin_1, abs=0.01)

    # Slot 2 : prix=20, gas=0, batterie neutre (0 MW) -> pas de cout.
    expected_revenue_2 = 20 * (100 + 10 + 5) * 0.25
    expected_fuel_2 = (12 * 100 + 5 * 10) * 0.25
    assert results_df.loc[2, "revenue_eur"] == pytest.approx(expected_revenue_2, abs=0.01)
    assert results_df.loc[2, "fuel_cost_eur"] == pytest.approx(expected_fuel_2, abs=0.01)
    assert results_df.loc[2, "gas_startup_cost_eur"] == 0.0


def test_extract_results_separates_realized_from_forecast_price(mocked_df, mocked_variables):
    results_df = optimizer.extract_results(mocked_df, *mocked_variables)
    assert list(results_df["da_price"]) == [50.0, 150.0, 20.0]
    assert list(results_df["da_price_forecast"]) == [55.0, 140.0, 25.0]


def test_extract_results_reports_u_gas_and_start_gas_unchanged(mocked_df, mocked_variables):
    results_df = optimizer.extract_results(mocked_df, *mocked_variables)
    assert list(results_df["u_gas"]) == [0, 1, 0]
    assert list(results_df["start_gas"]) == [0.0, 1.0, 0.0]


def test_extract_results_handles_missing_demand_column_gracefully(mocked_df, mocked_variables):
    """Cas limite : demand_mw absente du DataFrame -> pas de colonne
    demand_mw en sortie, mais pas de crash (branche deja geree dans le
    code existant, verifiee ici explicitement)."""
    df_no_demand = mocked_df.drop(columns=["demand_mw"])
    results_df = optimizer.extract_results(df_no_demand, *mocked_variables)
    assert "demand_mw" not in results_df.columns
    assert len(results_df) == N
