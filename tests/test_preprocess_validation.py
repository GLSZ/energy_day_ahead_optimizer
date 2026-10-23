"""
Tests des cas limites de preprocess.validate_optimizer_input() --
demande explicite de la Correction 4 : valeurs manquantes, entrees
invalides, limites de production (pmin <= pmax).

Methode : DataFrame minimal construit directement (toutes les colonnes
critiques necessaires), pas tire du pipeline reel -- legitime pour
tester unitairement une fonction de validation pure, independamment des
donnees ENTSO-E/Open-Meteo.
"""
import pandas as pd
import pytest

import preprocess

N = 4


def _valid_df() -> pd.DataFrame:
    """DataFrame minimal qui satisfait tous les controles de
    validate_optimizer_input (sert de point de depart, chaque test ne
    casse qu'UNE seule invariante a la fois)."""
    return pd.DataFrame({
        "da_price_eur_mwh": [50.0] * N,
        "da_price_forecast_eur_mwh": [55.0] * N,
        "gas_marginal_cost": [93.0] * N,
        "nuclear_marginal_cost": [12.0] * N,
        "nuclear_pmin": [300.0] * N,
        "nuclear_pmax": [1500.0] * N,
        "gas_pmin": [0.0] * N,
        "gas_pmax": [500.0] * N,
        "hydro_pmin": [0.0] * N,
        "hydro_pmax": [300.0] * N,
        "wind_pmin": [0.0] * N,
        "wind_pmax": [50.0] * N,
        "solar_pmin": [0.0] * N,
        "solar_pmax": [30.0] * N,
        "battery_pmin": [-100.0] * N,
        "battery_pmax": [100.0] * N,
        "demand_mw": [40000.0] * N,
        "hydro_inflow_hm3": [0.01] * N,
    })


def test_valid_dataframe_passes_without_raising():
    preprocess.validate_optimizer_input(_valid_df())  # ne doit lever aucune exception


def test_missing_critical_column_raises_keyerror():
    df = _valid_df().drop(columns=["da_price_forecast_eur_mwh"])
    with pytest.raises(KeyError, match="da_price_forecast_eur_mwh"):
        preprocess.validate_optimizer_input(df)


def test_nan_in_critical_column_raises_valueerror():
    df = _valid_df()
    df.loc[1, "nuclear_marginal_cost"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        preprocess.validate_optimizer_input(df)


@pytest.mark.parametrize("pmin_col,pmax_col", [
    ("nuclear_pmin", "nuclear_pmax"),
    ("gas_pmin", "gas_pmax"),
    ("hydro_pmin", "hydro_pmax"),
    ("wind_pmin", "wind_pmax"),
    ("solar_pmin", "solar_pmax"),
    ("battery_pmin", "battery_pmax"),
])
def test_pmin_greater_than_pmax_raises_valueerror(pmin_col, pmax_col):
    """Limite de production : pmin ne doit jamais depasser pmax, pour
    aucun des 6 actifs."""
    df = _valid_df()
    df.loc[0, pmin_col] = df.loc[0, pmax_col] + 1.0
    with pytest.raises(ValueError, match="Contrainte violée"):
        preprocess.validate_optimizer_input(df)


def test_negative_hydro_inflow_raises_valueerror():
    df = _valid_df()
    df.loc[2, "hydro_inflow_hm3"] = -0.01
    with pytest.raises(ValueError, match="négatifs"):
        preprocess.validate_optimizer_input(df)


def test_missing_hydro_inflow_column_is_tolerated():
    """hydro_inflow_hm3 n'est pas dans critical_cols -- son absence ne
    doit pas faire planter la validation (verifie le comportement
    actuel explicitement)."""
    df = _valid_df().drop(columns=["hydro_inflow_hm3"])
    preprocess.validate_optimizer_input(df)  # ne doit pas lever
