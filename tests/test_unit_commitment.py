"""
Tests pour l'unit commitment minimal du gaz (Correction 3) :
  - centrale a l'arret (prix toujours defavorable -> U_gas=0, P_gas=0)
  - demarrage detecte et facture (transition 0 -> 1), y compris au slot 0
  - fonctionnement continu (un seul demarrage facture, jamais un par slot)
  - impossibilite de produire hors des limites liees a U_gas

Hypotheses validees avec l'utilisateur (voir discussion) et rappelees ici :
  - Etat avant le slot 0 = ETEINT (u_initial=0). Un premier test avait
    montre qu'en ignorant cette transition, le solveur pouvait toujours
    eviter tout cout de demarrage en se declarant "allume" gratuitement
    des minuit (faille economique, pas un bug) -> corrige en comptant
    explicitement un demarrage si U_gas[0]=1.
  - Pmin technique = 0 quand allume (pas de palier de charge minimal).
  - Duree minimale de marche/arret : non implementee (hors perimetre).

Methode : on part du vrai jeu de donnees deja present dans le repo
(optimizer_input_2026-09-13.csv), puis on CONSTRUIT une trajectoire de
prix deliberement contrastee (tres bas puis tres haut) sur la colonne de
prevision, pour forcer un cycle marche/arret observable. Ce n'est pas
une donnee fictive injectee dans le pipeline reel : c'est une
perturbation controlee, locale au test.
"""
import datetime

import pulp
import pytest

import config
import optimizer

KNOWN_DATE = datetime.date(2026, 9, 13)


@pytest.fixture
def real_optimizer_input():
    config.TARGET_DATE = KNOWN_DATE
    return optimizer.load_optimizer_input()


def _build_and_solve(df, price_column="da_price_forecast_eur_mwh"):
    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = optimizer.build_lp_model(
        df, price_column=price_column
    )
    status = pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))]
    return status, prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas


def test_gas_stays_off_when_always_unprofitable(real_optimizer_input):
    """
    Prix toujours tres en dessous du SRMC gaz (~93 €/MWh) -> U_gas=0 et
    P_gas=0 partout. Avec la faille du slot 0 fermee, il n'y a plus
    aucune raison economique de jamais allumer le gaz ici.
    """
    df = real_optimizer_input.copy()
    df["da_price_forecast_eur_mwh"] = 0.0

    status, prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = _build_and_solve(df)
    assert status == "Optimal"

    for t in optimizer.SLOTS:
        assert (pulp.value(U_gas[t]) or 0.0) == 0.0, (
            f"U_gas[{t}] devrait etre 0 (aucun benefice a demarrer, et "
            "demarrer coute desormais toujours quelque chose)"
        )
        assert (pulp.value(P["gas"][t]) or 0.0) == pytest.approx(0.0, abs=1e-6)


def test_gas_startup_is_detected_and_costed(real_optimizer_input):
    """
    Prix defavorable sur les 40 premiers slots puis tres favorable ensuite
    -> le gaz doit demarrer une fois (detecte par Start_gas), et le cout
    de demarrage doit apparaitre exactement sur ce slot de transition.
    """
    df = real_optimizer_input.copy()
    prices = [0.0] * 40 + [500.0] * (optimizer.N_SLOTS - 40)
    df["da_price_forecast_eur_mwh"] = prices

    status, prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = _build_and_solve(df)
    assert status == "Optimal"

    u_values = {t: round(pulp.value(U_gas[t]) or 0.0) for t in optimizer.SLOTS}
    # NOTE : on ne peut pas exiger U_gas=0 sur TOUT le creneau 0-39 : la
    # contrainte de rampe preexistante (Ramp_up_gas, 50 MW/slot) force un
    # positionnement anticipe realiste (le gaz peut commencer a monter en
    # puissance quelques slots avant que le prix devienne favorable, pour
    # atteindre sa pleine capacite pile au bon moment). Verifie a la place
    # qu'il ne demarre pas des le slot 0 (pas de raison de le faire) et
    # qu'il est bien allume pendant la periode favorable.
    assert u_values[0] == 0, "le gaz ne devrait pas demarrer des le slot 0 sans raison"
    assert any(u_values[t] == 1 for t in range(40, optimizer.N_SLOTS)), (
        "le gaz ne demarre jamais alors que le prix est tres favorable (500 €/MWh)"
    )

    results_df = optimizer.extract_results(df, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas)

    n_starts = (results_df["start_gas"] > 0.5).sum()
    assert n_starts == 1, f"attendu exactement 1 demarrage, obtenu {n_starts}"

    total_startup_cost = results_df["gas_startup_cost_eur"].sum()
    gas_startup_cost_cfg = optimizer.ASSETS[optimizer.ASSET_CONFIG_KEY["gas"]]["startup_cost"]
    assert total_startup_cost == pytest.approx(gas_startup_cost_cfg, abs=1.0)

    # Coherence : la marge reportee doit bien avoir deduit ce cout.
    total_margin = results_df["margin_eur"].sum()
    total_revenue_minus_fuel = (results_df["revenue_eur"] - results_df["fuel_cost_eur"]).sum()
    assert total_margin == pytest.approx(total_revenue_minus_fuel, abs=0.1)


def test_startup_cost_is_charged_at_slot_zero_if_gas_runs_immediately(real_optimizer_input):
    """
    Hypothese validee (revisee) : l'etat avant minuit est ETEINT -> si le
    gaz tourne des le slot 0, un cout de demarrage DOIT etre facture à
    t=0, exactement comme pour n'importe quel autre slot de transition.
    """
    df = real_optimizer_input.copy()
    df["da_price_forecast_eur_mwh"] = 500.0  # tres favorable toute la journee

    status, prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = _build_and_solve(df)
    assert status == "Optimal"
    assert (pulp.value(U_gas[0]) or 0.0) == 1.0, "le gaz devrait tourner des le slot 0 (tres favorable)"

    results_df = optimizer.extract_results(df, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas)

    gas_startup_cost_cfg = optimizer.ASSETS[optimizer.ASSET_CONFIG_KEY["gas"]]["startup_cost"]
    assert results_df.loc[0, "gas_startup_cost_eur"] == pytest.approx(gas_startup_cost_cfg)
    assert results_df.loc[0, "start_gas"] == pytest.approx(1.0)


def test_continuous_operation_has_exactly_one_startup(real_optimizer_input):
    """
    Si le gaz tourne en continu toute la journee (favorable partout), un
    seul demarrage doit etre facture au total (au slot 0), jamais un par
    slot ensuite tant qu'il reste allume.
    """
    df = real_optimizer_input.copy()
    df["da_price_forecast_eur_mwh"] = 500.0

    status, prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = _build_and_solve(df)
    assert status == "Optimal"

    results_df = optimizer.extract_results(df, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas)
    n_starts = (results_df["start_gas"] > 0.5).sum()
    assert n_starts == 1, f"attendu exactement 1 demarrage sur la journee, obtenu {n_starts}"


def test_capacity_link_forbids_production_above_zero_when_off(real_optimizer_input):
    """
    Verification structurelle directe de la contrainte de liaison
    Gas_UC_capacity_link_t : P_gas[t] <= capacity_max * U_gas[t].
    Si on force U_gas[t]=0 manuellement, P_gas[t] doit etre contraint a 0,
    quel que soit le prix (meme tres favorable).
    """
    df = real_optimizer_input.copy()
    df["da_price_forecast_eur_mwh"] = 500.0

    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = optimizer.build_lp_model(df)

    # On force U_gas a 0 sur tous les slots (simule une indisponibilite totale)
    for t in optimizer.SLOTS:
        prob += (U_gas[t] == 0, f"force_off_{t}")

    status = pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))]
    assert status == "Optimal"

    for t in optimizer.SLOTS:
        assert (pulp.value(P["gas"][t]) or 0.0) == pytest.approx(0.0, abs=1e-6), (
            f"P_gas[{t}] > 0 alors que U_gas[{t}] est force a 0 -> la "
            "contrainte de liaison ne fonctionne pas"
        )
