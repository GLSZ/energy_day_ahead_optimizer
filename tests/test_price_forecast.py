"""
Tests pour forecast_price.py (Correction 2) :
  - absence de fuite temporelle (aucune date >= target_date utilisee)
  - fenetre de lookback (exclusion d'un historique trop ancien / hors
    regime de marche, ex: le releve isole de 2015)
  - comportement explicite en cas d'historique insuffisant (pas de
    valeur inventee, erreur levee)
  - mesure de qualite (MAE/MAPE) non triviale, jamais presumee bonne

Donnees utilisees : les vrais CSV de data/raw/ deja presents dans le
repo (pas de donnees synthetiques).
"""
import datetime

import pandas as pd
import pytest

import forecast_price as fp


KNOWN_DATE = datetime.date(2026, 9, 13)       # historique disponible avant
EARLIEST_DATE = datetime.date(2026, 4, 26)    # rien de disponible avant (sous 365j)


def test_no_future_leakage_in_listed_files():
    """
    Pour target_date=2026-09-13, des fichiers posterieurs existent bien
    sur disque (2026-09-15, 2026-09-29, 2026-10-xx). Ce test verifie
    qu'aucun d'eux n'est jamais selectionne.
    """
    files = fp._list_historical_price_files(KNOWN_DATE)
    assert files, "aucun historique trouve alors qu'il devrait y en avoir"
    for file_date, _path in files:
        assert file_date < KNOWN_DATE, (
            f"fuite temporelle : {file_date} >= {KNOWN_DATE} a ete inclus "
            "dans l'historique utilise pour la prevision"
        )


def test_no_future_leakage_in_loaded_history():
    """Meme verification mais sur les donnees reellement chargees/concatenees."""
    history = fp.load_historical_prices(KNOWN_DATE)
    assert not history.empty
    assert (history["source_date"] < KNOWN_DATE).all(), (
        "la table d'historique charge contient une date >= target_date"
    )


def test_lookback_window_excludes_old_regime():
    """
    Le fichier data/raw/da_prices_..._2015-05-22_dates.csv existe mais
    est a plus de 365 jours avant 2026-09-13 -> doit etre exclu par
    defaut (hypothese documentee : eviter de melanger des regimes de
    marche tres differents).
    """
    history = fp.load_historical_prices(KNOWN_DATE, lookback_days=365)
    assert datetime.date(2015, 5, 22) not in set(history["source_date"])

    # Avec une fenetre volontairement tres large, elle doit reapparaitre.
    history_wide = fp.load_historical_prices(KNOWN_DATE, lookback_days=5000)
    assert datetime.date(2015, 5, 22) in set(history_wide["source_date"])


def test_insufficient_history_raises_explicit_error():
    """
    2026-04-26 est la date la plus ancienne disponible dans data/raw/ ;
    avec la fenetre par defaut (365j), aucun historique n'existe avant
    elle -> doit lever InsufficientHistoryError, jamais fabriquer une
    valeur par defaut silencieuse.
    """
    with pytest.raises(fp.InsufficientHistoryError):
        fp.forecast_day_ahead_price(EARLIEST_DATE)


def test_forecast_has_no_missing_values_on_known_date():
    forecast = fp.forecast_day_ahead_price(KNOWN_DATE)
    assert len(forecast) == 96
    assert forecast.isna().sum() == 0


def test_quality_metric_is_computed_and_not_trivially_zero():
    """
    Une moyenne historique simple ne doit jamais etre presentee comme
    parfaite : sur une date volatile reelle, l'erreur mesuree doit etre
    strictement positive (sinon la mesure elle-meme serait suspecte).
    """
    forecast = fp.forecast_day_ahead_price(KNOWN_DATE)
    quality = fp.evaluate_forecast_quality(KNOWN_DATE, forecast)
    assert quality is not None
    assert quality["mae_eur_mwh"] > 0
    assert quality["n_slots"] == 96


def test_quality_metric_returns_none_without_ground_truth():
    """
    Si aucun prix realise n'existe pour la date (cas d'une vraie date
    future), on ne peut pas evaluer la prevision -> None, pas d'erreur
    fabriquee.
    """
    fake_future_date = datetime.date(2099, 1, 1)
    dummy_forecast = pd.Series([50.0] * 96, name="da_price_forecast_eur_mwh")
    quality = fp.evaluate_forecast_quality(fake_future_date, dummy_forecast)
    assert quality is None
