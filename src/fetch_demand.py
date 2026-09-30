# src/fetch_demand.py

"""
fetch_demand.py — Extraction de la prévision de charge système depuis ENTSO-E.

RTE publie chaque jour avant 13h00 la prévision de charge pour le lendemain
(J-1 forecast), avec une résolution de 15 minutes depuis octobre 2025.
C'est exactement la donnée qu'un trader Day-Ahead utilise pour contraindre
son plan de dispatch : on ne peut vendre que ce que le réseau consomme.

Sources :
  - query_load_forecast() → prévision de charge publiée par RTE (J-1)
  - query_load()          → charge réelle (pour vérification a posteriori)

Retourne une pd.Series de 96 valeurs (pas 15min) en MW.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_DATE, PRICE_ZONE

#env_path = Path(__file__).parent.parent / "logs.env"
env_path = Path(__file__).parent / "logs.env"
load_dotenv(env_path)


# ─────────────────────────────────────────────
# AUTHENTIFICATION
# ─────────────────────────────────────────────

def get_client() -> EntsoePandasClient:
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Token ENTSO-E manquant. "
            "Vérifie logs.env → ENTSOE_API_KEY=ton_token"
        )
    return EntsoePandasClient(api_key=api_key)


# ─────────────────────────────────────────────
# EXTRACTION PRÉVISION DE CHARGE (J-1)
# ─────────────────────────────────────────────

def fetch_load_forecast(
    target_date=TARGET_DATE,
    price_zone=PRICE_ZONE,
) -> pd.Series:
    """
    Récupère la prévision de charge publiée par RTE pour target_date.

    ENTSO-E Transparency Platform — processus métier :
      - RTE publie la prévision de charge J-1 avant 13h00 chaque jour
      - Résolution : 15min depuis octobre 2025, horaire avant
      - Unité : MW (mégawatts)
      - La prévision couvre minuit→minuit du jour cible

    Pourquoi utiliser la prévision J-1 et non la charge réelle ?
      → En conditions réelles, le trader soumet ses offres DA
        la veille pour le lendemain. Il n'a accès qu'à la prévision,
        pas à la charge réelle (qui n'existe pas encore).
      → Utiliser la charge réelle serait du look-ahead bias.
      → Exception : si target_date est dans le passé ET qu'on veut
        faire du backtesting, on peut utiliser query_load() à la place.

    Retourne une pd.Series de 96 valeurs indexées 0..95 (slots 15min).
    """
    client = get_client()

    start = pd.Timestamp(target_date, tz="UTC")
    end   = pd.Timestamp(target_date + timedelta(days=1), tz="UTC")

    print(f"[ENTSO-E] Prévision de charge J-1 — zone : {price_zone} — date : {target_date}")

    try:
        raw: pd.Series = client.query_load_forecast(
            country_code=price_zone,
            start=start,
            end=end,
        )

        # ENTSO-E retourne parfois un DataFrame → extraire la colonne
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]   # prend la première colonne "Forecasted Load"

    except Exception as e:
        raise RuntimeError(
            f"Échec query_load_forecast : {e}\n"
            "Causes possibles :\n"
            "  - Date trop ancienne (prévisions disponibles ~2 ans)\n"
            "  - Date dans le futur (prévision pas encore publiée)\n"
            "  - Zone de prix incorrecte"
        )

    # ── Nettoyage ─────────────────────────────────────────────────────────
    raw.index = raw.index.tz_convert("Europe/Paris")

    # Filtre sur le jour cible (sécurité DST)
    mask  = raw.index.date == target_date
    forecast = raw[mask].copy()

    n = len(forecast)
    if n == 0:
        raise ValueError(
            f"Aucune donnée de prévision pour {target_date}.\n"
            "La prévision J-1 est disponible environ 18h avant le jour cible."
        )

    # ── Harmonisation vers 96 slots 15min ─────────────────────────────────
    # Avant oct. 2025 : données horaires (24 valeurs) → upsampler en 15min
    # Après oct. 2025 : données 15min (96 valeurs) → utiliser directement
    if n == 24 or n == 23 or n == 25:
        print(f"[ENTSO-E] Données horaires détectées ({n}h) → upsampling 15min")
        forecast = _upsample_to_15min(forecast, target_date)
    elif n in range(80, 101):
        print(f"[ENTSO-E] Données 15min détectées ({n} slots)")
        if n != 96:
            # Reconstruit l'index datetime pour pouvoir interpoler
            start = pd.Timestamp(target_date, tz="Europe/Paris")
            end   = pd.Timestamp(
                target_date + timedelta(days=1), tz="Europe/Paris"
            )
            full_index    = pd.date_range(
                start=start, end=end,
                freq="15min", inclusive="left"
            )
            forecast.index = full_index[:n]          # assigne un index datetime
            forecast       = _pad_to_96_slots(forecast, target_date)
    else:
        raise ValueError(f"Nombre de valeurs inattendu : {n}")

    forecast.index = range(len(forecast))
    forecast.name  = "load_forecast_mw"

    # ── Validation ────────────────────────────────────────────────────────
    _validate_load(forecast)

    print(f"[ENTSO-E] Prévision OK — {len(forecast)} slots")
    print(f"          Min : {forecast.min():,.0f} MW  |  "
          f"Max : {forecast.max():,.0f} MW  |  "
          f"Moy : {forecast.mean():,.0f} MW")

    return forecast

def _pad_to_96_slots(series: pd.Series, target_date) -> pd.Series:
    """
    Complète une série incomplète à exactement 96 slots par
    réindexage sur l'index 15min complet + interpolation linéaire.

    Cas typiques :
      - 88 slots : 8 slots manquants en fin de journée
      - 92 slots : changement d'heure (heure d'été → heure d'hiver)
      - 84 slots : données partielles publiées tôt le matin
    """
    from datetime import timedelta

    start = pd.Timestamp(target_date, tz="Europe/Paris")
    end   = pd.Timestamp(target_date + timedelta(days=1), tz="Europe/Paris")

    # Index 15min complet de la journée
    full_index = pd.date_range(
        start=start, end=end,
        freq="15min", inclusive="left"
    )

    # Réindexe + interpolation linéaire sur les slots manquants
    padded = (
        series
        .reindex(series.index.union(full_index))
        .interpolate(method="linear")
        .reindex(full_index)
    )

    n_filled = len(full_index) - len(series)
    if n_filled > 0:
        print(f"[PAD] {n_filled} slots interpolés pour compléter à 96")

    return padded

def fetch_load_actual(
    target_date=TARGET_DATE,
    price_zone=PRICE_ZONE,
) -> pd.Series:
    """
    Récupère la charge réelle pour target_date (a posteriori).

    Utilisé pour :
      - Vérifier la qualité de la prévision J-1 après coup
      - Backtesting avec données réelles (en acceptant le look-ahead)
      - Comparaison prévision vs réel dans visualize.py

    Même logique de nettoyage que fetch_load_forecast().
    """
    client = get_client()

    start = pd.Timestamp(target_date, tz="UTC")
    end   = pd.Timestamp(target_date + timedelta(days=1), tz="UTC")

    print(f"[ENTSO-E] Charge réelle — zone : {price_zone} — date : {target_date}")

    try:
        raw: pd.Series = client.query_load(
            country_code=price_zone,
            start=start,
            end=end,
        )

        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]

    except Exception as e:
        raise RuntimeError(f"Échec query_load : {e}")

    raw.index = raw.index.tz_convert("Europe/Paris")
    mask  = raw.index.date == target_date
    actual = raw[mask].copy()

    n = len(actual)
    if n in (23, 24, 25):
        actual = _upsample_to_15min(actual, target_date)
    #elif n not in range(92, 101):
        #raise ValueError(f"Nombre de valeurs inattendu pour la charge réelle : {n}")
    elif n in range(80, 101):
        print(f"[ENTSO-E] Données 15min détectées ({n} slots)")
        if n != 96:
            # Construit l'index 15min complet — nécessaire pour _pad_to_96_slots
            start_ts   = pd.Timestamp(target_date, tz="Europe/Paris")
            end_ts     = pd.Timestamp(target_date + timedelta(days=1), tz="Europe/Paris")
            full_index = pd.date_range(
                start=start_ts, end=end_ts,
                freq="15min", inclusive="left"
            )
            actual.index = full_index[:n]
            actual       = _pad_to_96_slots(actual, target_date)
    elif n not in range(80, 101):
        raise ValueError(f"Nombre de valeurs inattendu pour la charge réelle : {n}")


    actual.index = range(len(actual))
    actual.name  = "load_actual_mw"

    _validate_load(actual)

    print(f"[ENTSO-E] Charge réelle OK — {len(actual)} slots")
    print(f"          Min : {actual.min():,.0f} MW  |  "
          f"Max : {actual.max():,.0f} MW  |  "
          f"Moy : {actual.mean():,.0f} MW")

    return actual


# ─────────────────────────────────────────────
# UPSAMPLING HORAIRE → 15MIN
# ─────────────────────────────────────────────

def _upsample_to_15min(
    series: pd.Series,
    target_date,
) -> pd.Series:
    """
    Convertit une série horaire (24h) en série 15min (96 slots)
    par interpolation linéaire.

    Méthode : interpolation linéaire entre deux valeurs horaires.
    Ex : 09h=45000 MW, 10h=47000 MW
      → 09h00=45000, 09h15=45500, 09h30=46000, 09h45=46500

    Pourquoi linéaire et pas step (maintien de la valeur) ?
      La charge électrique varie continûment — une interpolation
      linéaire représente mieux la réalité qu'un échelon.
      Pour un usage dans l'optimiseur, la différence est faible
      car les rampes des actifs lissent de toute façon la production.
    """
    # Reconstruit un index datetime propre en Europe/Paris
    start = pd.Timestamp(target_date, tz="Europe/Paris")
    end   = pd.Timestamp(target_date + timedelta(days=1), tz="Europe/Paris")

    # Index horaire original
    hourly_index = pd.date_range(start=start, end=end, freq="h", inclusive="left")

    # Réindexe sur l'index horaire pour être sûr de l'alignement
    series_clean = pd.Series(
        series.values[:len(hourly_index)],
        index=hourly_index,
        name=series.name,
    )

    # Index 15min cible
    index_15min = pd.date_range(start=start, end=end, freq="15min", inclusive="left")

    # Réindexe + interpolation linéaire
    upsampled = (
        series_clean
        .reindex(series_clean.index.union(index_15min))
        .interpolate(method="time")
        .reindex(index_15min)
    )

    return upsampled


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def _validate_load(series: pd.Series) -> None:
    """
    Contrôles de cohérence sur la courbe de charge.

    Plages typiques pour la France (RTE) :
      - Minimum absolu : ~25 000 MW (nuit d'été, dimanche)
      - Maximum absolu : ~102 000 MW (vague de froid, janvier)
      - Variations intra-journalières : ±30% autour de la moyenne
    """
    LOAD_MIN_MW = 20_000    # MW — en dessous : données suspectes
    LOAD_MAX_MW = 120_000   # MW — au-dessus : données suspectes

    #print("DEBUG type:", type(series))
    #print("DEBUG shape:", series.shape)
    #print("DEBUG columns:", getattr(series, "columns", None))

    # Sécurité : force en Series si DataFrame
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    if series.isnull().any():
        n_null = series.isnull().sum()
        raise ValueError(f"{n_null} valeurs NaN dans la courbe de charge")

    if series.min() < LOAD_MIN_MW:
        print(f"[WARN] Charge anormalement basse : {series.min():,.0f} MW "
              f"(seuil : {LOAD_MIN_MW:,} MW)")

    if series.max() > LOAD_MAX_MW:
        print(f"[WARN] Charge anormalement haute : {series.max():,.0f} MW "
              f"(seuil : {LOAD_MAX_MW:,} MW)")

    # Vérifie la cohérence du profil journalier
    # La charge nocturne (00h-05h) doit être < la charge diurne (08h-20h)
    night_slots = list(range(0, 20))     # 00h00 → 05h00
    day_slots   = list(range(32, 80))    # 08h00 → 20h00

    avg_night = series.iloc[night_slots].mean()
    avg_day   = series.iloc[day_slots].mean()

    if avg_night > avg_day:
        print(f"[WARN] Profil journalier inversé : "
              f"nuit {avg_night:,.0f} MW > jour {avg_day:,.0f} MW — "
              f"vérifie la cohérence des données")


# ─────────────────────────────────────────────
# MARGE DE SÉCURITÉ
# ─────────────────────────────────────────────

def add_security_margin(
    load_forecast: pd.Series,
    margin_pct: float = 0.05,
) -> pd.Series:
    """
    Ajoute une marge de sécurité sur la prévision de charge.

    En pratique, RTE impose aux producteurs de prévoir une réserve
    opérationnelle au-dessus de la charge prévue pour couvrir :
      - Les aléas de prévision de la demande (~2-3%)
      - Les réserves primaire et secondaire (~1-2%)

    Ici on modélise ça simplement comme un pourcentage additionnel.
    La marge standard en France est ~5% (réglage via config).

    Paramètre
    ---------
    margin_pct : float — marge en fraction (0.05 = 5%)

    Retourne la série de charge avec la marge intégrée.
    """
    load_with_margin = load_forecast * (1 + margin_pct)
    load_with_margin.name = "load_forecast_with_margin_mw"

    print(f"[MARGIN] Marge de sécurité : +{margin_pct*100:.0f}%")
    print(f"         Charge max prévue  : {load_forecast.max():,.0f} MW")
    print(f"         Charge max marginée: {load_with_margin.max():,.0f} MW")

    return load_with_margin


# ─────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────

def save_load(
    forecast: pd.Series,
    actual: pd.Series = None,
    output_dir: str = "data/raw",
) -> str:
    """
    Sauvegarde la prévision (et éventuellement la charge réelle)
    dans data/raw/ au format CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    forecast_col = forecast.copy()
    forecast_col.name = "load_forecast_mw"   # ← nom fixe dans le CSV

    # df = forecast.to_frame()
    df = forecast_col.to_frame()

    if actual is not None:
        # Aligne la charge réelle sur le même index
        actual_aligned = actual.reindex(forecast.index)
        df["load_actual_mw"] = actual_aligned

        # Calcule l'erreur de prévision si les deux sont disponibles
        df["forecast_error_mw"]  = df["load_forecast_mw"] - df["load_actual_mw"]
        df["forecast_error_pct"] = (
            df["forecast_error_mw"] / df["load_actual_mw"] * 100
        )
        mape = df["forecast_error_pct"].abs().mean()
        print(f"[QUALITY] MAPE prévision vs réel : {mape:.2f}%")

    filepath = os.path.join(output_dir, f"load_{TARGET_DATE}.csv")
    df.to_csv(filepath, header=True)
    print(f"[SAVE] Charge sauvegardée → {filepath}")

    return filepath


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_fetch_demand(
    target_date=TARGET_DATE,
    fetch_actual: bool = True,
    margin_pct: float = 0.05,
) -> dict:
    """
    Pipeline complet :
      fetch_forecast → (fetch_actual) → add_margin → save

    Paramètres
    ----------
    fetch_actual : bool — récupère aussi la charge réelle pour comparaison
    margin_pct   : float — marge de sécurité opérationnelle

    Retourne un dict :
    {
      "forecast"          : pd.Series — prévision brute ENTSO-E
      "forecast_margined" : pd.Series — prévision + marge de sécurité
      "actual"            : pd.Series ou None — charge réelle
    }
    """
    print("=" * 60)
    print(f"FETCH DEMAND — {target_date}")
    print("=" * 60)

    # Prévision J-1
    forecast = fetch_load_forecast(target_date)

    # Charge réelle (optionnel — peut échouer si date trop récente)
    actual = None
    if fetch_actual:
        try:
            actual = fetch_load_actual(target_date)
        except Exception as e:
            print(f"[WARN] Charge réelle indisponible : {e}")
            print("       Normal si target_date = aujourd'hui ou demain")

    # Marge de sécurité
    forecast_margined = add_security_margin(forecast, margin_pct)

    # Sauvegarde
    save_load(forecast_margined, actual)

    print("=" * 60)
    print("FETCH DEMAND TERMINÉ")
    print("=" * 60)

    return {
        "forecast"          : forecast,
        "forecast_margined" : forecast_margined,
        "actual"            : actual,
    }


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = run_fetch_demand()

    print("\nPrévision de charge — 8 premiers slots :")
    print(result["forecast"].head(8).to_string())

    print("\nAvec marge de sécurité — 8 premiers slots :")
    print(result["forecast_margined"].head(8).to_string())

    if result["actual"] is not None:
        print("\nCharge réelle — 8 premiers slots :")
        print(result["actual"].head(8).to_string())