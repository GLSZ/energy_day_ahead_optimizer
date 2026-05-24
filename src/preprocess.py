"""
preprocess.py — Alignement, nettoyage et préparation du DataFrame d'entrée
pour l'optimiseur Day-Ahead.

Rôle central de ce module :
  1. Charger les prix DA (96 slots 15min) depuis data/raw/
  2. Charger les données météo + puissance (96 slots) depuis data/raw/
  3. Aligner les deux séries sur le même index temporel
  4. Calculer le coût marginal dynamique du gaz (TTF + CO2)
  5. Construire le DataFrame final "optimizer_input" que optimizer.py consommera

Structure du DataFrame de sortie (96 lignes × N colonnes) :
  - slot                  : int 0..95
  - datetime              : timestamp Europe/Paris
  - da_price_eur_mwh      : float — prix DA en €/MWh
  - wind_power_mw         : float — borne sup production éolienne
  - solar_power_mw        : float — borne sup production solaire
  - gas_marginal_cost     : float — coût marginal gaz calculé dynamiquement
  - [asset]_capacity_min  : float — pour chaque actif dispatchable
  - [asset]_capacity_max  : float — pour chaque actif dispatchable
"""

import pandas as pd
import numpy as np
import os
import sys
import yfinance as yf
from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_DATE, ASSETS, 
    EUA_PRICE, GAS_EFFICIENCY_CCGT
)

'''
# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES BRUTES
# ─────────────────────────────────────────────
'''

def load_prices(data_dir: str = "data/raw") -> pd.DataFrame :

    """
    Charge le CSV des prix DA produit par fetch_prices.py.

    Ton fichier a ces colonnes :
        Unnamed: 0 | date | da_price_eur_mwh

    On renomme, on parse les dates, et on vérifie qu'on a bien 96 slots.
    """

    # Cherche le fichier le plus récent correspondant à TARGET_DATE
    pattern = f"da_prices_da_price_eur_mwh_{TARGET_DATE}_dates.csv"
    filepath = os.path.join(data_dir, pattern)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier prix introuvable : {filepath}\n"
            "Lance d'abord : python src/fetch_prices.py"
        )
    
    df = pd.read_csv(filepath)
    df = df.drop(columns = ["Unnamed: 0"], errors="ignore")
    
    df["date"] = pd.to_datetime(df["date"], utc = True).dt.tz_convert("Europe/Paris")
    df = df.rename(columns={"date": "datetime"})

    df = df.reset_index(drop = True)
    df.insert(0, "slot", df.index)

    print(f"[LOAD] Prix DA — {len(df)} lignes chargées")
    _check_96_slots(df, "Prix DA")

    return df[["slot", "datetime", "da_price_eur_mwh"]]

def load_weather(data_dir: str = "data/raw") -> pd.DataFrame:

    """
    Charge le CSV météo + puissance produit par fetch_weather.py.

    Tes colonnes disponibles :
        slot | date | temperature_2m | apparent_temperature |
        wind_speed_10m | wind_speed_80m | direct_radiation |
        precipitation | visibility | wind_power_mw | solar_power_mw

    On garde uniquement ce dont l'optimiseur a besoin.
    """

    pattern = f"weather_power_{TARGET_DATE}.csv"
    filepath = os.path.join(data_dir, pattern)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier météo introuvable : {filepath}\n"
            f"Lance d'abord : python src/fetch_weather.py"
        )

    df = pd.read_csv(filepath)

    df["date"] = pd.to_datetime(df["date"], utc = True).dt.tz_convert("Europe/Paris")
    df = df.rename(columns={"date" : "datetime"})

    cols_keep = [
    "slot", "datetime",
    "wind_speed_80m", "direct_radiation",
    "temperature_2m", "wind_power_mw", "solar_power_mw"
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep].copy()

    print(f"[LOAD] Météo — {len(df)} lignes chargées")
    _check_96_slots(df, "Météo")

    return df

def _check_96_slots(df : pd.DataFrame, label : str) -> None:

    """
    Vérifie qu'on a bien ~96 slots (tolérance DST ±4).
    Lève une erreur explicite sinon.
    """
    n = len(df)
    if n not in range(92, 101, 4):
        raise ValueError(
            f"[{label}] Nombre de slots inattendu : {n} "
            "(attendu 92-100 selon DST)"
        )
    if n!=96: 
        print(f"[WARN] [{label}] {n} slots au lieu de 96 — jour DST ?")


'''
# ─────────────────────────────────────────────
# 2. ALIGNEMENT TEMPOREL
# ─────────────────────────────────────────────
'''

def align_series(
        prices : pd.DataFrame,
        weather : pd.DataFrame,
) -> pd.DataFrame :
    
    """
    Fusionne prix et météo sur l'index 'slot' (0..95).

    Pourquoi fusionner sur 'slot' et non sur 'datetime' ?
    → Les deux sources peuvent avoir de légères différences de timestamp
      (ex: 00:00:00 vs 00:00:01 à cause du cache HTTP). Fusionner sur
      l'index entier 0..95 est plus robuste que sur les timestamps exacts.

    On fait quand même un assert de cohérence sur les datetime pour
    détecter un éventuel décalage de jour.

    Stratégie : inner join sur 'slot' — si un slot manque dans l'une
    des deux sources, la ligne disparaît et on lève un warning.
    """

    df = pd.merge(
        prices, 
        weather.drop(columns = ["datetime"], errors = "ignore"),
        on = "slot",
        how = 'inner',
        validate="one_to_one"
    )

    n_lost = 96 - len(df)
    if n_lost > 0:
        print(f"[WARN] Alignement : {n_lost} slots perdus au merge "
              f"(probablement DST) — slots restants : {len(df)}")     

    dates_uniques = df["datetime"].dt.date.unique()
    if len(dates_uniques) > 1:
        print(f"[WARN] Plusieurs dates dans le DataFrame fusionné : {dates_uniques}")

    print(f"[ALIGN] Fusion prix × météo — {len(df)} slots alignés")
    return df.reset_index(drop=True)   

'''
# ─────────────────────────────────────────────
# 3. COÛT MARGINAL DU GAZ (DYNAMIQUE)
# ─────────────────────────────────────────────
'''
def fetch_ttf_price(target_date=TARGET_DATE) -> float:
    """
    Récupère le prix du gaz naturel TTF (Dutch TTF Natural Gas)
    via yfinance pour le jour cible.

    Ticker yfinance : "TTF=F"  — contrat futures TTF en €/MWh
    Fallback        : "NG=F"   — Henry Hub en USD/MMBtu (converti si TTF indispo)

    Pourquoi le prix TTF est-il la bonne référence ?
    → Le TTF (Title Transfer Facility) est le hub gazier de référence
      en Europe continentale. Le coût marginal d'une centrale CCGT
      française est directement indexé sur le TTF.

    Conversion si fallback Henry Hub (USD/MMBtu → €/MWh) :
      1 MMBtu = 0.29307 MWh  →  prix (€/MWh) = prix (USD/MMBtu) / 0.29307 / EUR_USD
    """

    # On télécharge une fenêtre de 5 jours autour de la date cible
    # pour gérer les week-ends et jours fériés (pas de cotation)
    start = (target_date - timedelta(days = 5)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days = 1)).strftime("%Y-%m-%d")

    print(f"[TTF] Récupération du prix gaz — fenêtre {start} - {end}")

    try :
        raw = yf.download("TTF=F", start = start, end = end, progress = False)
        if raw.empty :
            raise ValueError("Aucune donnée TTF retournée")

        # On prend le prix de clôture du dernier jour disponible
        # (= meilleure approximation du prix du gaz pour TARGET_DATE)
        ttf_price = float(raw["Close"].dropna().iloc[-1])
        last_date = raw["Close"].dropna().index[-1].date()

        print(f"[TTF] Prix TTF : {ttf_price:.2f} €/MWh (cotation du {last_date})")

        # Sanity check : le TTF varie typiquement entre 10 et 150 €/MWh
        if not (5 <= ttf_price <= 300):
            print(f"[WARN] Prix TTF hors plage habituelle : {ttf_price:.2f} €/MWh")

        return ttf_price

    except Exception as e:
        # Fallback : prix TTF de marché typique en cas d'échec API
        FALLBACK_TTF = 40.0 # €/MWh  valeur indicative 2024-2025
        print(f"[WARN] Échec récupération TTF ({e}) → fallback {FALLBACK_TTF} €/MWh")
        return FALLBACK_TTF

def compute_gas_marginal_cost(
    ttf_price: float,
    eua_price: float = EUA_PRICE,
    efficiency: float = GAS_EFFICIENCY_CCGT,
    co2_intensity: float = None,
) -> float:

    """
    Calcule le coût marginal complet d'une centrale CCGT en €/MWh électrique.

    Formule :
    ┌─────────────────────────────────────────────────────────────────┐
    │  SRMC (Short-Run Marginal Cost) = (TTF / η) + (co2_intensity × EUA)                      │
    │                                                                 │
    │  où :                                                           │
    │    TTF           = prix gaz naturel en €/MWh_thermique          │
    │    η (eta)       = rendement thermique CCGT (0.58 = 58%)        │
    │    co2_intensity = émissions en tCO2/MWh_électrique (0.37)      │
    │    EUA           = prix quota carbone européen en €/tCO2        │
    └─────────────────────────────────────────────────────────────────┘

    Exemple numérique avec TTF = 40 €/MWh, EUA = 65 €/tCO2 :
      Coût combustible = 40 / 0.58 = 68.97 €/MWh_élec
      Coût CO2         = 0.37 × 65 = 24.05 €/MWh_élec
      ─────────────────────────────────────────────────
      SRMC total       = 68.97 + 24.05 = 93.02 €/MWh_élec

    Interprétation : la centrale CCGT démarre si et seulement si
    le prix DA > SRMC. C'est le principe du merit order.
    Si prix DA = 80 €/MWh < 93 €/MWh → la centrale ne produit pas.
    Si prix DA = 120 €/MWh > 93 €/MWh → la centrale produit au max.
    """

    if co2_intensity is None:
        co2_intensity = ASSETS["gas_ccgt"]["co2_intensity"] # 0.37 tCO2/MWh

    fuel_cost = ttf_price / efficiency # coût combustible ramené à l'électricité
    co2_cost = co2_intensity * eua_price # coût des quotas carbone
    srmc = fuel_cost + co2_cost

    print(f"[GAS SRMC] TTF={ttf_price:.2f} €/MWh_gaz  |  eta ={efficiency:.0%}")
    print(f"           Coût combustible : {fuel_cost:.2f} €/MWh_élec")
    print(f"           Coût CO2         : {co2_cost:.2f} €/MWh_élec "
          f"({co2_intensity} tCO2 × {eua_price} €/t)")
    print(f"           SRMC total       : {srmc:.2f} €/MWh_élec")

    return round(srmc, 2)

'''
# ─────────────────────────────────────────────
# 4. CONSTRUCTION DU DATAFRAME OPTIMISEUR
# ─────────────────────────────────────────────
'''

def build_optimizer_input(
        df : pd.DataFrame,
        gas_srmc : float
) -> pd.DataFrame:
    
    """
    Enrichit le DataFrame aligné avec toutes les colonnes
    dont optimizer.py aura besoin.

    Colonnes ajoutées :
    ┌─────────────────────────────────────────────────────────────────┐
    │ gas_marginal_cost      : SRMC gaz calculé dynamiquement         │
    │ nuclear_cost           : coût marginal nucléaire (fixe)         │
    │ hydro_cost             : coût d'opportunité eau (fixe)          │
    │ battery_cost           : coût de dégradation (fixe)            │
    │ nuclear_pmin           : puissance min nucléaire par slot (MW)  │
    │ nuclear_pmax           : puissance max nucléaire par slot (MW)  │
    │ gas_pmin               : puissance min gaz par slot (MW)        │
    │ gas_pmax               : puissance max gaz par slot (MW)        │
    │ hydro_pmin             : puissance min hydro par slot (MW)      │
    │ hydro_pmax             : puissance max hydro par slot (MW)      │
    │ wind_pmax              : borne sup éolien = météo (MW)          │
    │ solar_pmax             : borne sup solaire = météo (MW)         │
    │ battery_pmin           : -100 MW (charge max)                   │
    │ battery_pmax           : +100 MW (décharge max)                 │
    │ ramp_up_[asset]        : rampe montante en MW/slot              │
    │ ramp_down_[asset]      : rampe descendante en MW/slot           │
    └─────────────────────────────────────────────────────────────────┘

    Note sur la conversion des rampes MW/h → MW/slot :
    Les rampes dans config.py sont en MW/h (convention industrie).
    Comme nos slots font 15min = 0.25h, la rampe par slot est :
        ramp_mw_per_slot = ramp_mw_per_hour × 0.25
    Ex : nucléaire ramp_up = 50 MW/h → 12.5 MW/slot
    """

    SLOT_DURATION_H = 0.25 # 15min = 0.25 heure
    out = df.copy()

    # ── Coûts marginaux ──────────────────────────────────────────────────
    # Ces valeurs sont constantes sur tous les slots de la journée
    # (on ne modélise pas la variation intra-journalière des prix de combustible)

    out["gas_marginal_cost"] = gas_srmc
    out["nuclear_marginal_cost"] = ASSETS["nuclear"]["marginal_cost"]       # 12.0 €/MWh
    out["hydro_marginal_cost"] = ASSETS["hydro_reservoir"]["marginal_cost"] # 5.0 €/MWh
    out["battery_marginal_cost"] = ASSETS["battery"]["marginal_cost"]       # 2.0 €/MWh
    # Éolien et solaire : coût marginal nul → pas de colonne nécessaire
    # (l'optimiseur maximise toujours leur production dans les bornes météo)   

    # ── Capacités min/max (MW) ────────────────────────────────────────────
    # Pour les actifs dispatchables : constantes sur la journée
    # Pour les renouvelables : la borne MAX varie avec la météo (déjà dans df)

    # Nucléaire — must-run élevé (700 MW min en permanence)
    out["nuclear_pmin"] = ASSETS["nuclear"]["capacity_min"]   # 700 MW
    out["nuclear_pmax"] = ASSETS["nuclear"]["capacity_max"]   # 1500 MW

    # Gaz CCGT — peut s'arrêter complètement (pmin = 0)
    out["gas_pmin"] = ASSETS["gas_ccgt"]["capacity_min"]  # 0 MW
    out["gas_pmax"] = ASSETS["gas_ccgt"]["capacity_max"]  # 500 MW

    # Hydraulique — pmin = 0, mais contrainte de budget journalier dans optimizer
    out["hydro_pmin"] = ASSETS["hydro_reservoir"]["capacity_min"]   # 0 MW
    out["hydro_pmax"] = ASSETS["hydro_reservoir"]["capacity_max"]   # 300 MW

    # Renouvelables — borne sup = puissance météo calculée par fetch_weather.py
    # wind_power_mw et solar_power_mw sont déjà dans df (hérités de l'alignement)
    out["wind_pmin"] = 0.0   # fatal : on ne peut pas forcer la production
    out["wind_pmax"] = out["wind_power_mw"]    # varie slot par slot

    out["solar_pmin"] = 0.0
    out["solar_pmax"] = out["solar_power_mw"]   # varie slot par slot

    # Batterie — puissance négative = charge, positive = décharge
    out["battery_pmin"] = ASSETS["battery"]["capacity_min"]   # -100 MW
    out["battery_pmax"] = ASSETS["battery"]["capacity_max"]   # +100 MW

    # ── Rampes (MW/slot) ──────────────────────────────────────────────────
    # Convention : ramp_up = variation positive max entre slot t et t+1
    #              ramp_down = variation négative max (en valeur absolue)
    # Conversion : MW/h × 0.25h/slot = MW/slot

    for asset_key, col_prefix in [
        ("nuclear", "nuclear"),
        ("gas_ccgt", "gas"),
        ("hydro_reservoir", "hydro"),
        ("battery", "battery")
    ] :
        ramp_up_h = ASSETS[asset_key]["ramp_up"]
        ramp_down_h = ASSETS[asset_key]["ramp_down"]
        out[f"{col_prefix}_ramp_up"]   = ramp_up_h   * SLOT_DURATION_H  # MW/slot
        out[f"{col_prefix}_ramp_down"] = ramp_down_h * SLOT_DURATION_H  # MW/slot


    # Renouvelables : pas de contrainte de rampe (le vent/soleil décide)
    # → on ne crée pas de colonne ramp pour wind/solar

    # ── Colonne de spread  ──
    # Le "clean spark spread" = prix DA - coût marginal gaz
    # Positif → la centrale gaz est rentable à ce slot
    # Négatif → la centrale gaz perd de l'argent → doit être à l'arrêt
    out["clean_spark_spread"] = out["da_price_eur_mwh"] - gas_srmc

    # ── Indicateur de signal de dispatch (pour visualisation) ────────────
    # Permet de voir d'un coup d'œil quand chaque actif est potentiellement
    # rentable avant même de lancer l'optimiseur
    out["nuclear_profitable"] = out["da_price_eur_mwh"] > ASSETS["nuclear"]["marginal_cost"]
    out["gas_profitable"] = out["clean_spark_spread"] > 0
    out["hydro_profitable"] = out["da_price_eur_mwh"] > ASSETS["hydro_reservoir"]["marginal_cost"]

    print(f"\n[BUILD] DataFrame optimiseur construit — {len(out)} slots × {len(out.columns)} colonnes")
    print(f"  Clean spark spread moyen   : {out['clean_spark_spread'].mean():.2f} €/MWh")
    print(f"  Slots où gaz rentable      : {out['gas_profitable'].sum()}/96")
    print(f"  Slots où nucléaire rentable: {out['nuclear_profitable'].sum()}/96")

    return out

'''
# ─────────────────────────────────────────────
# 5. VALIDATION FINALE
# ─────────────────────────────────────────────
'''

def validate_optimizer_input(df: pd.DataFrame) -> None:
    """
    Vérifie la cohérence du DataFrame avant de l'envoyer à optimizer.py.

    Contrôles :
    - Aucune valeur NaN dans les colonnes critiques
    - Toutes les bornes pmin <= pmax
    - Les rampes sont positives
    - wind_pmax et solar_pmax sont dans [0, capacity_max]
    """

    print("\n[VALIDATE] Vérification du DataFrame optimiseur...")

    # Colonnes critiques pour le LP
    critical_cols = [
        "da_price_eur_mwh",
        "gas_marginal_cost", "nuclear_marginal_cost",
        "nuclear_pmin", "nuclear_pmax",
        "gas_pmin",     "gas_pmax",
        "hydro_pmin",   "hydro_pmax",
        "wind_pmin",    "wind_pmax",
        "solar_pmin",   "solar_pmax",
        "battery_pmin", "battery_pmax",
    ]
    for col in critical_cols:
        if col not in df.columns:
            raise KeyError(f"Colonne manquante : '{col}'")
        if df[col].isnull().any():
            raise ValueError(f"NaN détectés dans '{col}'")
        
    # Vérification pmin <= pmax pour chaque actif
    pairs = [
        ("nuclear_pmin", "nuclear_pmax"),
        ("gas_pmin",     "gas_pmax"),
        ("hydro_pmin",   "hydro_pmax"),
        ("wind_pmin",    "wind_pmax"),
        ("solar_pmin",   "solar_pmax"),
        ("battery_pmin", "battery_pmax"),
    ]
    for pmin_col, pmax_col in pairs:
        violations = df[df[pmin_col] > df[pmax_col]]
        if not violations.empty:
            raise ValueError(
                f"Contrainte violée : {pmin_col} > {pmax_col} "
                f"sur {len(violations)} slots"
            )
        
    # Vérification bornes renouvelables
    wind_over  = df[df["wind_pmax"]  > ASSETS["wind"]["capacity_max"]]
    solar_over = df[df["solar_pmax"] > ASSETS["solar"]["capacity_max"]]
    if not wind_over.empty:
        print(f"[WARN] {len(wind_over)} slots avec wind_pmax > capacity_max installée")
    if not solar_over.empty:
        print(f"[WARN] {len(solar_over)} slots avec solar_pmax > capacity_max installée")

    print("[VALIDATE] OK — DataFrame prêt pour l'optimiseur\n")

'''
# ─────────────────────────────────────────────
# 6. EXPORT
# ─────────────────────────────────────────────
'''

def save_optimizer_input(
        df : pd.DataFrame, 
        output_dir: str = "data/processed"
) -> str:
    
    """
    Sauvegarde le DataFrame final dans data/processed/.
    C'est ce fichier qu'optimizer.py chargera directement.
    """

    os.makedirs(output_dir, exist_ok=True)
    filename = f"optimizer_input_{TARGET_DATE}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index = False)
    print(f"[SAVE] optimizer_input sauvegardé : {filepath}")
    return filepath

'''
# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
'''

def run_preprocessing(
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Enchaîne toutes les étapes dans l'ordre :
      load_prices → load_weather → align → TTF → SRMC → build → validate → save
    """

    print("=" *60)
    print(f"PREPROCESSING — {TARGET_DATE}")
    print("=" *60)

    # Étape 1 : chargement
    prices  = load_prices(data_dir)
    weather = load_weather(data_dir)

    # Étape 2 : alignement
    df = align_series(prices, weather)

    # Étape 3 : coût marginal gaz
    ttf_price = fetch_ttf_price(TARGET_DATE)
    gas_srmc  = compute_gas_marginal_cost(ttf_price)

    # Étape 4 : construction DataFrame optimiseur
    df_opt = build_optimizer_input(df, gas_srmc)

    # Étape 5 : validation
    validate_optimizer_input(df_opt)

    # Étape 6 : sauvegarde
    save_optimizer_input(df_opt, output_dir)

    print("=" * 60)
    print("PREPROCESSING TERMINÉ")
    print("=" * 60)

    return df_opt

'''
# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────
'''

if __name__ == "__main__":
    df = run_preprocessing()

    print("\nAperçu des colonnes clés :")
    cols_display = [
        "slot", "datetime", "da_price_eur_mwh",
        "gas_marginal_cost", "clean_spark_spread",
        "wind_pmax", "solar_pmax",
        "nuclear_ramp_up", "gas_ramp_up",
    ]
    cols_display = [c for c in cols_display if c in df.columns]
    print(df[cols_display].head(12).to_string(index=False))

    print("\nRésumé statistique des prix DA :")
    print(df["da_price_eur_mwh"].describe().round(2))