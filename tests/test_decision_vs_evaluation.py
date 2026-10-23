"""
Verifie la separation decision / evaluation, ET les deux scenarios
cote a cote demandes explicitement (voir discussion) :

  - "forecast"          : decision = prevision (da_price_forecast_eur_mwh)
                          -> dispatch realiste, avant cloture d'enchere.
  - "perfect_foresight" : decision = prix realise (da_price_eur_mwh)
                          -> dispatch a connaissance parfaite, legitime
                          pour une decision prise APRES cloture
                          d'enchere, ou comme borne theorique.

Dans les deux cas, extract_results() evalue TOUJOURS le P&L sur le
prix REALISE (ce qui a vraiment ete paye/percu sur le marche).

Methode : on part d'un vrai jeu de donnees deja present dans le repo
(optimizer_input_2026-09-13.csv), puis on PERTURBE une copie en memoire
du seul champ 'da_price_forecast_eur_mwh' avec une valeur constante et
facilement identifiable, en laissant 'da_price_eur_mwh' (reel) intact.
Ceci n'est pas une donnee fictive injectee dans le pipeline : c'est une
perturbation controlee, locale au test, qui sert uniquement a prouver
sans ambiguite QUELLE colonne alimente l'objectif.
"""
import datetime

import pulp
import pytest

import config
import optimizer

KNOWN_DATE = datetime.date(2026, 9, 13)

DISTINCT_FORECAST_PRICE = 777.0  # valeur impossible a confondre avec un prix reel


@pytest.fixture
def real_optimizer_input():
    config.TARGET_DATE = KNOWN_DATE
    return optimizer.load_optimizer_input()


def _objective_coefficient(prob, variable):
    """Coefficient d'une variable dans l'objectif, par identite d'objet
    (voir note PuLP dans tests/test_hydro_balance.py : __eq__ est
    surcharge sur LpVariable, 'in'/'==' directs donnent des faux
    positifs)."""
    for v, coeff in prob.objective.items():
        if v is variable:
            return coeff
    return None


def test_objective_uses_forecast_price_not_realized(real_optimizer_input):
    df = real_optimizer_input.copy()

    real_price_slot0 = df["da_price_eur_mwh"].iloc[0]
    assert real_price_slot0 != DISTINCT_FORECAST_PRICE, (
        "le prix reel du slot 0 coincide par hasard avec la valeur de "
        "test -> changer DISTINCT_FORECAST_PRICE"
    )

    df["da_price_forecast_eur_mwh"] = DISTINCT_FORECAST_PRICE

    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = optimizer.build_lp_model(df)

    nuclear_marginal_cost = df["nuclear_marginal_cost"].iloc[0]
    expected_coeff_nuclear = (
        (DISTINCT_FORECAST_PRICE - nuclear_marginal_cost) * optimizer.SLOT_DURATION_H
    )

    coeff = _objective_coefficient(prob, P["nuclear"][0])
    assert coeff is not None, "P_nuclear_0 n'apparait pas dans l'objectif"
    assert coeff == pytest.approx(expected_coeff_nuclear), (
        f"coefficient objectif = {coeff}, attendu {expected_coeff_nuclear} "
        "(calcule a partir de la PREVISION) -> l'objectif lit-il encore "
        "le prix realise par erreur ?"
    )


def test_objective_can_switch_to_realized_price_perfect_foresight(real_optimizer_input):
    """
    Scenario 'perfect_foresight' : price_column="da_price_eur_mwh" doit
    faire lire le prix REALISE a l'objectif (legitime pour une decision
    prise apres cloture d'enchere / comme borne theorique).
    """
    df = real_optimizer_input.copy()

    real_price_slot0 = df["da_price_eur_mwh"].iloc[0]
    nuclear_marginal_cost = df["nuclear_marginal_cost"].iloc[0]
    expected_coeff_nuclear = (
        (real_price_slot0 - nuclear_marginal_cost) * optimizer.SLOT_DURATION_H
    )

    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = optimizer.build_lp_model(
        df, price_column="da_price_eur_mwh"
    )

    coeff = _objective_coefficient(prob, P["nuclear"][0])
    assert coeff is not None
    assert coeff == pytest.approx(expected_coeff_nuclear), (
        f"coefficient objectif = {coeff}, attendu {expected_coeff_nuclear} "
        "(calcule a partir du prix REALISE)"
    )


def test_extract_results_keeps_realized_price_for_pnl(real_optimizer_input):
    """
    extract_results() doit continuer a calculer revenue_eur/fuel_cost_eur
    a partir du prix REALISE, meme quand la prevision est tres differente
    (sinon le P&L reporte ne refleterait pas ce qui a reellement ete
    gagne/perdu sur le marche).
    """
    df = real_optimizer_input.copy()
    df["da_price_forecast_eur_mwh"] = DISTINCT_FORECAST_PRICE

    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = optimizer.build_lp_model(df)
    status = pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))]
    assert status == "Optimal"

    results_df = optimizer.extract_results(df, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas)

    # Le P&L (revenue_eur) doit etre coherent avec le prix REALISE
    # (da_price), pas avec la prevision factice de 777 €/MWh.
    assert "da_price_forecast" in results_df.columns
    assert (results_df["da_price_forecast"] == DISTINCT_FORECAST_PRICE).all()
    assert not (results_df["da_price"] == DISTINCT_FORECAST_PRICE).any(), (
        "da_price (realise) a ete contamine par la prevision factice"
    )
