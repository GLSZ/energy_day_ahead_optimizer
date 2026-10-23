# src/forecast_price.py

"""
forecast_price.py — Prévision du prix Day-Ahead utilisée pour la DÉCISION
de dispatch (fonction objectif du LP dans optimizer.py).

Distinction fondamentale avec fetch_prices.py :
  - fetch_prices.py   -> prix RÉALISÉ (post-clôture du marché ENTSO-E),
                         utilisé uniquement pour évaluer le P&L a
                         posteriori (extract_results dans optimizer.py).
  - forecast_price.py -> prix PRÉVU, seule donnée dont un opérateur
                         day-ahead dispose réellement au moment de
                         soumettre ses offres. C'est ce prix qui doit
                         alimenter la fonction objectif du LP.

Méthode retenue : moyenne historique par (slot de 15min, jour de
semaine), calculée UNIQUEMENT à partir des prix réalisés de dates
STRICTEMENT antérieures à la date cible. Le filtre de date est appliqué
sur le nom de fichier avant toute lecture de contenu — aucune fuite
temporelle n'est possible par construction.

Repli à 3 paliers (en fonction de la richesse réelle de l'historique
disponible sur disque, qui est très épars dans ce projet — quelques
dates ponctuelles, pas une série continue) :
  A. moyenne par (slot, jour de semaine) — le plus précis
  B. moyenne par slot seul (tous jours de semaine confondus)
  C. moyenne par heure (slot // 4), tous jours confondus — le plus
     grossier, utilisé seulement si A et B manquent de données

Hypothèses et limites assumées explicitement :
  - `DEFAULT_LOOKBACK_DAYS = 365` : on ignore l'historique plus ancien
    qu'un an pour éviter de mélanger des régimes de marché très
    différents (ex : un relevé isolé de 2015 pour prévoir 2026).
  - `MIN_OBS_PER_BUCKET = 2` : une maille avec moins de 2 observations
    historiques n'est pas jugée assez fiable, on redescend d'un palier.
  - Jours fériés : NON traités (pas de calendrier férié dans ce projet,
    aucune dépendance nouvelle ajoutée sans validation préalable).
  - C'est une moyenne historique simple, pas un modèle prédictif : sa
    qualité DOIT être mesurée (voir evaluate_forecast_quality) et ne
    doit jamais être présumée performante.
  - Si aucun historique n'existe du tout avant la date cible, on lève
    une erreur explicite plutôt que d'inventer une valeur par défaut.
"""

import os
import re
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PRICE_FILENAME_RE = re.compile(
    r"^da_prices_da_price_eur_mwh_(\d{4}-\d{2}-\d{2})_dates\.csv$"
)

DEFAULT_LOOKBACK_DAYS = 365
MIN_OBS_PER_BUCKET = 2


class InsufficientHistoryError(RuntimeError):
    """
    Levée quand aucun historique de prix réalisé n'est disponible
    strictement avant la date cible. Ne JAMAIS attraper cette exception
    pour la remplacer silencieusement par une valeur inventée.
    """


# ─────────────────────────────────────────────
# 1. DÉCOUVERTE DES FICHIERS HISTORIQUES DISPONIBLES
# ─────────────────────────────────────────────

def _list_historical_price_files(
    target_date: date,
    raw_dir: str = "data/raw",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list:
    """
    Liste les fichiers de prix réalisés STRICTEMENT antérieurs à
    target_date, dans la fenêtre [target_date - lookback_days, target_date[.

    Le filtre de date est fait sur le NOM du fichier (pas son contenu),
    donc aucune lecture de donnée "future" ne se produit jamais, même
    par accident.
    """
    if not os.path.isdir(raw_dir):
        return []

    window_start = target_date - timedelta(days=lookback_days)
    matches = []
    for filename in os.listdir(raw_dir):
        m = RAW_PRICE_FILENAME_RE.match(filename)
        if not m:
            continue
        file_date = date.fromisoformat(m.group(1))
        if window_start <= file_date < target_date:
            matches.append((file_date, os.path.join(raw_dir, filename)))

    matches.sort(key=lambda x: x[0])
    return matches


def _load_one_historical_file(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier de prix réalisé au même format que
    preprocess.load_prices(), mais sans dépendre de TARGET_DATE global
    (on charge un fichier précis, potentiellement une date passée
    quelconque de l'historique).
    """
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Paris")
    df = df.rename(columns={"date": "datetime"})
    df = df.reset_index(drop=True)
    df.insert(0, "slot", df.index)
    return df[["slot", "datetime", "da_price_eur_mwh"]]


def load_historical_prices(
    target_date: date,
    raw_dir: str = "data/raw",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Charge et concatène tous les prix réalisés disponibles strictement
    avant target_date (dans la fenêtre de lookback). Ajoute les colonnes
    `source_date` et `weekday` (0=lundi .. 6=dimanche) nécessaires au
    regroupement par maille.

    Retourne un DataFrame vide (pas une erreur) si aucun fichier ne
    correspond — c'est forecast_day_ahead_price() qui décide quoi faire
    de l'absence totale d'historique.
    """
    files = _list_historical_price_files(target_date, raw_dir, lookback_days)
    if not files:
        return pd.DataFrame(
            columns=["slot", "datetime", "da_price_eur_mwh", "source_date", "weekday"]
        )

    frames = []
    for file_date, filepath in files:
        df = _load_one_historical_file(filepath)
        df["source_date"] = file_date
        df["weekday"] = file_date.weekday()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 2. PRÉVISION PAR MOYENNE HISTORIQUE À PALIERS
# ─────────────────────────────────────────────

def forecast_day_ahead_price(
    target_date: date,
    raw_dir: str = "data/raw",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_obs_per_bucket: int = MIN_OBS_PER_BUCKET,
    n_slots: int = 96,
) -> pd.Series:
    """
    Construit une prévision de prix DA pour target_date, slot par slot
    (0..n_slots-1), à partir de l'historique réalisé strictement
    antérieur à target_date.

    Lève InsufficientHistoryError si aucun historique n'est disponible
    du tout dans la fenêtre — ne fabrique jamais de valeur par défaut.
    """
    history = load_historical_prices(target_date, raw_dir, lookback_days)

    if history.empty:
        raise InsufficientHistoryError(
            f"Aucun historique de prix DA disponible strictement avant "
            f"{target_date} (fenêtre de {lookback_days} jours) dans "
            f"'{raw_dir}'. Impossible de produire une prévision sans "
            "données réelles — lancez d'abord fetch_prices.py sur des "
            "dates antérieures, ou réduisez lookback_days si une donnée "
            "plus ancienne existe mais est hors fenêtre."
        )

    target_weekday = target_date.weekday()
    n_dates = history["source_date"].nunique()

    tier_a = (
        history[history["weekday"] == target_weekday]
        .groupby("slot")["da_price_eur_mwh"]
        .agg(["mean", "count"])
    )
    tier_b = history.groupby("slot")["da_price_eur_mwh"].agg(["mean", "count"])

    history_hourly = history.copy()
    history_hourly["hour"] = history_hourly["slot"] // 4
    tier_c = history_hourly.groupby("hour")["da_price_eur_mwh"].agg(["mean", "count"])

    forecast = {}
    tier_used = {}
    for t in range(n_slots):
        if t in tier_a.index and tier_a.loc[t, "count"] >= min_obs_per_bucket:
            forecast[t] = tier_a.loc[t, "mean"]
            tier_used[t] = "A_slot_weekday"
        elif t in tier_b.index and tier_b.loc[t, "count"] >= min_obs_per_bucket:
            forecast[t] = tier_b.loc[t, "mean"]
            tier_used[t] = "B_slot_only"
        else:
            hour = t // 4
            if hour in tier_c.index and tier_c.loc[hour, "count"] >= 1:
                forecast[t] = tier_c.loc[hour, "mean"]
                tier_used[t] = "C_hour_only"
            else:
                # Ne devrait arriver que si l'historique est tronqué
                # (ex: tous les fichiers dispo sont des jours DST à 92
                # slots et ne couvrent pas ce slot précis).
                raise InsufficientHistoryError(
                    f"Aucune observation historique disponible pour le "
                    f"slot {t} (heure {hour}) avant {target_date}, même "
                    "en regroupant par heure seule."
                )

    forecast_series = pd.Series(forecast, name="da_price_forecast_eur_mwh").sort_index()

    tiers_count = pd.Series(tier_used).value_counts().to_dict()
    print(
        f"[FORECAST] Prévision prix DA pour {target_date} — "
        f"{n_dates} date(s) historique(s) utilisée(s) "
        f"(fenêtre {lookback_days}j, strictement < {target_date})"
    )
    print(f"[FORECAST] Répartition des mailles utilisées : {tiers_count}")
    if n_dates < 5 or tiers_count.get("C_hour_only", 0) > 0:
        print(
            f"[FORECAST][WARN] Historique peu fourni ({n_dates} date(s)) "
            "-> prévision peu fiable, à ne pas présumer performante. "
            "Voir evaluate_forecast_quality()."
        )

    return forecast_series


# ─────────────────────────────────────────────
# 3. MESURE DE QUALITÉ (A POSTERIORI UNIQUEMENT)
# ─────────────────────────────────────────────

def evaluate_forecast_quality(
    target_date: date,
    forecast_series: pd.Series,
    raw_dir: str = "data/raw",
) -> dict:
    """
    Compare la prévision au prix RÉELLEMENT réalisé pour target_date,
    si ce prix est disponible (utile en backtesting sur une date passée).

    Retourne None si le réalisé n'est pas disponible (ex : target_date
    est une vraie date future). Le réalisé n'est utilisé ICI que pour
    ÉVALUER la prévision après coup — jamais pour la construire.
    """
    pattern = f"da_prices_da_price_eur_mwh_{target_date}_dates.csv"
    filepath = os.path.join(raw_dir, pattern)
    if not os.path.exists(filepath):
        print(
            f"[FORECAST] Pas de prix réalisé disponible pour {target_date} "
            "-> qualité de la prévision non évaluable."
        )
        return None

    realized = _load_one_historical_file(filepath)["da_price_eur_mwh"]

    if len(realized) != len(forecast_series):
        print(
            f"[FORECAST][WARN] longueur réalisé ({len(realized)}) != "
            f"prévision ({len(forecast_series)}) — probablement un jour "
            "DST. Comparaison tronquée au plus petit des deux."
        )
        n = min(len(realized), len(forecast_series))
        realized = realized.iloc[:n]
        forecast_values = forecast_series.values[:n]
    else:
        forecast_values = forecast_series.values

    error = forecast_values - realized.values
    mae = float(abs(error).mean())

    # Le MAPE est numériquement instable sur les prix DA : ils passent
    # fréquemment par zéro ou deviennent négatifs (surplus renouvelable),
    # ce qui fait exploser le dénominateur |réalisé| vers 0 et produit
    # des pourcentages d'erreur absurdes sans rapport avec la qualité
    # réelle de la prévision. On exclut les slots où |réalisé| < 1 €/MWh
    # du calcul du MAPE (documenté, pas silencieux) et on privilégie le
    # MAE comme métrique principale pour ce marché.
    near_zero_mask = realized.abs().values < 1.0
    n_excluded = int(near_zero_mask.sum())
    if n_excluded < len(realized):
        safe_error = error[~near_zero_mask]
        safe_realized = realized.values[~near_zero_mask]
        mape = float((abs(safe_error) / abs(safe_realized)).mean() * 100)
    else:
        mape = None  # tous les slots ont un prix réalisé proche de 0

    print(
        f"[FORECAST] Qualité de la prévision (vs réalisé, a posteriori) — "
        f"MAE={mae:.2f} €/MWh  "
        f"MAPE={'N/A' if mape is None else f'{mape:.1f}%'} "
        f"({n_excluded} slot(s) exclu(s) du MAPE, |prix réalisé| < 1 €/MWh)  "
        f"(n={len(realized)} slots)"
    )
    print(
        "[FORECAST] Le MAE est la métrique la plus fiable pour ce marché "
        "(le MAPE est instable quand le prix réalisé est proche de zéro "
        "ou négatif, fréquent sur les marchés électriques)."
    )
    if mape is not None and mape > 30:
        print(
            "[FORECAST][WARN] MAPE élevé (>30%) — cette prévision par "
            "moyenne historique est peu performante sur cette date, à "
            "interpréter avec prudence."
        )

    return {
        "mae_eur_mwh": mae,
        "mape_pct": mape,
        "mape_n_excluded": n_excluded,
        "n_slots": len(realized),
    }
