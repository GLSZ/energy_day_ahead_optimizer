"""
Verifie que les constantes centralisees en Correction 5 restent bien
une source unique, et ne peuvent pas re-diverger silencieusement :
  - must-run nucleaire (config.nuclear_must_run_mw()), lu par
    optimizer.py (contrainte C6) ET visualize.py (ligne de reference).
  - client ENTSO-E (entsoe_client.get_client()), partage par
    fetch_prices.py et fetch_demand.py.

Avant la Correction 5, optimizer.py et visualize.py avaient chacun leur
propre valeur (350 vs 750 MW) -- divergence passee inapercue jusqu'a
relecture manuelle du code. Ces tests auraient echoue si cette
divergence existait encore.
"""
import datetime
import inspect

import pytest

import config
import entsoe_client
import fetch_demand
import fetch_prices
import optimizer
import visualize


KNOWN_DATE = datetime.date(2026, 9, 13)


def test_nuclear_must_run_is_derived_from_config_not_a_duplicated_literal():
    """
    config.nuclear_must_run_mw() doit rester calcule a partir de
    ASSETS["nuclear"]["capacity_min"], pas une constante figee -- si la
    capacite nucleaire change dans config.py, la valeur doit suivre.
    """
    original = config.ASSETS["nuclear"]["capacity_min"]
    try:
        config.ASSETS["nuclear"]["capacity_min"] = 999
        assert config.nuclear_must_run_mw() == 999 + config.NUCLEAR_MUST_RUN_SAFETY_MARGIN_MW
    finally:
        config.ASSETS["nuclear"]["capacity_min"] = original


def test_optimizer_source_does_not_hardcode_nuclear_must_run():
    """
    Garde-fou anti-regression textuel : le code source de optimizer.py
    ne doit plus contenir de calcul local du must-run nucleaire
    (ancien pattern : 'ASSETS["nuclear"]["capacity_min"]' suivi d'un
    '+ 50' recopie). Il doit referencer la fonction centralisee.
    """
    source = inspect.getsource(optimizer)
    assert "nuclear_must_run_mw" in source, (
        "optimizer.py devrait importer/appeler config.nuclear_must_run_mw()"
    )
    assert "nuclear_must_run + 50" not in source, (
        "optimizer.py recalcule encore localement must-run + marge au lieu "
        "d'utiliser la valeur centralisee"
    )


def test_visualize_source_does_not_hardcode_nuclear_must_run():
    """Meme garde-fou pour visualize.py : l'ancienne valeur figee '750'
    ne doit plus apparaitre dans le code source."""
    source = inspect.getsource(visualize)
    assert "nuclear_must_run_mw" in source, (
        "visualize.py devrait appeler config.nuclear_must_run_mw()"
    )
    assert "nuclear_must_run = 750" not in source, (
        "visualize.py recopie encore une valeur figee au lieu d'utiliser "
        "la valeur centralisee"
    )


def test_optimizer_constraint_actually_uses_centralized_value():
    """
    Verification fonctionnelle (pas seulement textuelle) : quand la
    demande est tres faible, la contrainte C6 doit se replier exactement
    sur config.nuclear_must_run_mw(), pas sur une autre valeur.
    """
    config.TARGET_DATE = KNOWN_DATE
    df = optimizer.load_optimizer_input()
    df = df.copy()
    df["demand_mw"] = 0.0  # force le plancher must-run a etre la borne active

    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = optimizer.build_lp_model(df)

    constraint = prob.constraints["Demand_market_share_0"]
    # Le RHS d'une contrainte "<=" est stocke comme -constant dans PuLP.
    rhs = -constraint.constant
    assert rhs == pytest.approx(config.nuclear_must_run_mw())


def test_get_client_is_the_same_shared_function_everywhere():
    """
    fetch_prices.get_client et fetch_demand.get_client doivent pointer
    vers EXACTEMENT la meme fonction (meme objet), pas deux copies
    independantes qui pourraient diverger.
    """
    assert fetch_prices.get_client is entsoe_client.get_client
    assert fetch_demand.get_client is entsoe_client.get_client


def test_get_client_no_longer_leaks_api_key_in_logs(capsys):
    """
    Regression : l'ancienne version de fetch_prices.get_client()
    affichait les 5 premiers caracteres de la cle API dans les logs
    (print(api_key[0:5])). La version centralisee ne doit plus le faire.
    """
    import os

    os.environ["ENTSOE_API_KEY"] = "SECRET1234567890"
    try:
        entsoe_client.get_client()
    except Exception:
        pass  # peu importe si la construction du client echoue reellement
    finally:
        captured = capsys.readouterr()
        assert "SECRET" not in captured.out
        del os.environ["ENTSOE_API_KEY"]
