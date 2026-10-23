"""
Tests unitaires pour compute_gas_marginal_cost (preprocess.py) —
demande explicite de la Correction 4.

Formule testee : SRMC = TTF/efficiency + co2_intensity * eua_price
(voir docstring de preprocess.compute_gas_marginal_cost).
"""
import pytest

import preprocess


def test_known_numeric_example_matches_docstring():
    """
    L'exemple numerique du docstring (TTF=40, EUA=65, eta=0.58, co2=0.37)
    donne SRMC ~= 93.02 €/MWh -- on verifie la valeur exacte, pas juste
    un ordre de grandeur.
    """
    srmc = preprocess.compute_gas_marginal_cost(
        ttf_price=40.0, eua_price=65.0, efficiency=0.58, co2_intensity=0.37
    )
    expected = round(40.0 / 0.58 + 0.37 * 65.0, 2)
    assert srmc == pytest.approx(expected)
    assert srmc == pytest.approx(93.02, abs=0.01)


def test_zero_carbon_price_removes_co2_term():
    """Avec eua_price=0, le SRMC se reduit au seul cout combustible."""
    srmc = preprocess.compute_gas_marginal_cost(
        ttf_price=40.0, eua_price=0.0, efficiency=0.58, co2_intensity=0.37
    )
    assert srmc == pytest.approx(round(40.0 / 0.58, 2))


def test_scales_linearly_with_ttf_price():
    """
    Le cout combustible est proportionnel au prix TTF (a l'arrondi
    pres : compute_gas_marginal_cost arrondit a 2 decimales, donc une
    tolerance de 0.01 € absorbe le double-arrondi plutot qu'un rel=
    tres strict qui le detecterait a tort comme une non-linearite).
    """
    srmc_1 = preprocess.compute_gas_marginal_cost(ttf_price=40.0, eua_price=0.0)
    srmc_2 = preprocess.compute_gas_marginal_cost(ttf_price=80.0, eua_price=0.0)
    assert srmc_2 == pytest.approx(srmc_1 * 2, abs=0.01)


def test_higher_efficiency_reduces_fuel_cost():
    """Un meilleur rendement thermique reduit le cout combustible (toutes
    choses egales par ailleurs)."""
    srmc_low_eff = preprocess.compute_gas_marginal_cost(
        ttf_price=40.0, eua_price=0.0, efficiency=0.50
    )
    srmc_high_eff = preprocess.compute_gas_marginal_cost(
        ttf_price=40.0, eua_price=0.0, efficiency=0.60
    )
    assert srmc_high_eff < srmc_low_eff


def test_uses_default_co2_intensity_from_config_when_not_provided():
    """Sans co2_intensity explicite, la valeur par defaut vient de
    ASSETS['gas_ccgt']['co2_intensity'] (0.37 au moment de l'ecriture)."""
    from config import ASSETS

    default_co2 = ASSETS["gas_ccgt"]["co2_intensity"]
    srmc_explicit = preprocess.compute_gas_marginal_cost(
        ttf_price=40.0, eua_price=65.0, efficiency=0.58, co2_intensity=default_co2
    )
    srmc_default = preprocess.compute_gas_marginal_cost(
        ttf_price=40.0, eua_price=65.0, efficiency=0.58
    )
    assert srmc_default == pytest.approx(srmc_explicit)


def test_result_is_rounded_to_two_decimals():
    srmc = preprocess.compute_gas_marginal_cost(ttf_price=41.234567, eua_price=65.0)
    assert srmc == round(srmc, 2)


def test_negative_ttf_price_is_not_silently_clamped():
    """
    Un prix TTF negatif est physiquement aberrant pour un cout
    combustible, mais compute_gas_marginal_cost ne fait aucune
    validation de plausibilite (contrairement a fetch_prices._validate_
    prices pour le prix DA) -- ce test documente ce comportement actuel
    plutot que d'en supposer un autre.
    """
    srmc = preprocess.compute_gas_marginal_cost(ttf_price=-10.0, eua_price=0.0)
    assert srmc < 0
