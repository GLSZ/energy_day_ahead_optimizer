import requests
import pandas as pd
import os
import sys
from datetime import timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_DATE, LOCATION, ASSETS

# ─────────────────────────────────────────────
# CONSTANTES DE CONVERSION
# ─────────────────────────────────────────────

WIND_CAPACITY_MW = ASSETS["wind"]["capacity_max"]
SOLAR_CAPACITY_MW = ASSETS["solar"]["capacity_max"]

STC_RADIATION   = 1000   # W/m² — irradiation standard (conditions de test panneau)
PANEL_EFFICIENCY = 0.18  # 18% — rendement typique panneau monocristallin

CUT_IN_KMH  = 10    # km/h — vitesse minimale de démarrage turbine
RATED_KMH   = 50    # km/h — vitesse à laquelle on atteint la puissance nominale
CUT_OUT_KMH = 110   # km/h — vitesse de coupure de sécurité (arrêt turbine)

# ─────────────────────────────────────────────
# APPEL API OPEN-METEO
# ─────────────────────────────────────────────

def _get_openmeteo_client() -> openmeteo_requests.Client:
    """
    Instancie le client Open-Meteo avec :
    - Cache disque 1h (.cache/) → évite de rappeler l'API si données fraîches
    - Retry x5 avec backoff exponentiel → robustesse réseau
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_raw_weather(
        target_date = TARGET_DATE,
        location = LOCATION,
) -> object : 
    
    """
    Appelle l'API Open-Meteo et retourne la réponse JSON brute.
    Open-Meteo est gratuite et ne nécessite aucune clé API.

    URL de base : https://api.open-meteo.com/v1/forecast

    Variables météo utiles à récupérer (paramètre 'hourly') :
    - windspeed_10m          : vitesse du vent à 10m (km/h)
    - windspeed_100m         : vitesse du vent à 100m — plus représentatif des éoliennes
    - shortwave_radiation    : irradiation solaire globale (W/m²)
    - temperature_2m         : température air (°C) — utile pour la demande
    - direct_radiation        : radiation directe (W/m²) — pour panneaux suiveurs

    Documentation complète : https://open-meteo.com/en/docs

    Retourne le JSON brut de l'API.
    """
    client = _get_openmeteo_client()

    start_str = target_date.strftime("%Y-%m-%d")
    end_str   = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude" : location["latitude"],
        "longitude" : location["longitude"], 
        "timezone" : location["timezone"], 
        "start_date" : start_str,
        "end_date" : end_str,
        #"start_minutely_15" : pd.Timestamp(target_date, tz="Europe/Paris"),#"UTC"),
        #"end_minutely_15" : pd.Timestamp(target_date + timedelta(days = 1), tz="Europe/Paris"),#"UTC"),
        "minutely_15" :  [
            "temperature_2m",
            "apparent_temperature", 
            "wind_speed_10m", #Instant	km/h (mph, m/s, knots)
            "wind_speed_80m", 
            "direct_radiation",
            "precipitation", #Preceding 15 minutes sum	mm (inch)
            "visibility"
        ], #Instant meters
        "hourly" : "direct_radiation"
    }

    print(f"[Open-Meteo] Appel API — {location['name']} — {target_date}")
    

    try : 
        responses = client.weather_api(
            BASE_URL,
            params = params)
    except Exception as e:
        raise RuntimeError(f"Échec appel Open-Meteo : {e}")
    
    response = responses[0]

    print(f"[Open-Meteo] OK — Lat: {response.Latitude():.4f}°N  "
          f"Lon: {response.Longitude():.4f}°E  "
          f"Élévation: {response.Elevation():.0f}m")

    return response

def parse_weather(
        response, 
        target_date = TARGET_DATE
) -> pd.DataFrame:
    
    """
    Transforme l'objet réponse Open-Meteo en DataFrame propre.

    L'API minutely_15 retourne potentiellement 2 jours (J et J+1).
    On filtre pour ne garder que les 96 pas de 15min du target_date.

    Colonnes retournées :
    - datetime        : timestamp Europe/Paris
    - slot            : int 0..95 (index du quart d'heure)
    - wind_speed_10m  : km/h
    - wind_speed_80m  : km/h
    - solar_radiation : W/m²  (= direct_radiation)
    - temperature     : °C
    """

    minutely = response.Minutely15()
    # Reconstruction de l'index datetime depuis les métadonnées de la réponse
    # Open-Meteo donne : Time() = timestamp Unix du premier point
    #                    TimeEnd() = timestamp Unix du dernier point
    #                    Interval() = pas en secondes (= 900s pour 15min)

    timestamps = pd.date_range(
        start = pd.Timestamp(minutely.Time(), unit = "s", tz = "UTC"),
        end = pd.Timestamp(minutely.TimeEnd(), unit = "s", tz = "UTC"),
        freq = pd.Timedelta(seconds = minutely.Interval()),
        inclusive = "left"
    ).tz_convert("Europe/Paris")

    # Extraction des variables dans l'ordre où elles ont été demandées
    # (l'index dans Variables(i) correspond à l'ordre de la liste "minutely_15")
    df = pd.DataFrame({
        "date"        : timestamps,
        "temperature_2m"  : minutely.Variables(0).ValuesAsNumpy(),
        "apparent_temperature" : minutely.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m"  : minutely.Variables(2).ValuesAsNumpy(),  # wind_speed_10m
        "wind_speed_80m"  : minutely.Variables(3).ValuesAsNumpy(),  # wind_speed_80m
        "direct_radiation" : minutely.Variables(4).ValuesAsNumpy(),  # direct_radiation
        "precipitation"     : minutely.Variables(5).ValuesAsNumpy(),  # temperature_2m
        "visibility"     : minutely.Variables(6).ValuesAsNumpy(), 
    })

    df = df[df["date"].dt.date == target_date].copy()
    df = df.reset_index(drop = True)
    df.insert(0, "slot", df.index)

    n = len(df) 
    if n not in (92, 96, 100):
        raise ValueError(
            f"Nombre de pas météo inattendu : {n} (attendu ~96 pour 15min)"
        )
    if n != 96:
        print(f"[WARN] Changement d'heure détecté : {n} pas pour le {target_date}")

    for col in ["wind_speed_10m", "wind_speed_80m", "direct_radiation"]:
        if df[col].isnull().any():
            n_null = df[col].isnull().sum()
            print(f"[WARN] {n_null} valeurs NaN dans '{col}' → interpolation linéaire")
            df[col] = df[col].interpolate(method="linear")
                        
    print(f"[Open-Meteo] Parsing OK — {n} pas de 15min récupérés")
    print(f"  Vent 80m   : min {df['wind_speed_80m'].min():.1f}  "
          f"max {df['wind_speed_80m'].max():.1f} km/h")
    print(f"  Radiation  : min {df['direct_radiation'].min():.1f}  "
          f"max {df['direct_radiation'].max():.1f} W/m²")
    print(f"  Température: min {df['temperature_2m'].min():.1f}  "
          f"max {df['temperature_2m'].max():.1f} °C")
    
    return df



def compute_wind_power(wind_speed_kmh : pd.Series) -> pd.Series:
    """
    Courbe de puissance éolienne simplifiée (power curve).

    Physique : la puissance extraite du vent est P = ½ ρ A v³ × Cp
    → puissance proportionnelle au CUBE de la vitesse entre cut-in et rated.

    Implémentation :
    ┌──────────────────────────────────────────────────────────┐
    │  v < cut_in  (10 km/h) : P = 0 MW  (turbine à l'arrêt) │
    │  cut_in ≤ v < rated    : P = Pmax × (v - cut_in)³       │
    │                              / (rated - cut_in)³         │
    │  rated ≤ v < cut_out   : P = Pmax  (puissance nominale) │
    │  v ≥ cut_out (110 km/h): P = 0 MW  (arrêt sécurité)    │
    └──────────────────────────────────────────────────────────┘

    Exemple concret avec WIND_CAPACITY_MW = 200 MW :
      v = 0  km/h → P = 0 MW   (sous cut-in)
      v = 10 km/h → P = 0 MW   (seuil exact cut-in, puissance nulle)
      v = 30 km/h → P = 200 × (30-10)³ / (50-10)³
                      = 200 × 8000 / 64000 = 25 MW
      v = 50 km/h → P = 200 MW (puissance nominale atteinte)
      v = 80 km/h → P = 200 MW (dans la plage nominale)
      v = 120km/h → P = 0 MW   (au-dessus du cut-out, arrêt)
    """

    WIND_CAPACITY_MW = ASSETS["wind"]["capacity_max"]
    CUT_IN_KMH  = 10     # vitesse minimale de démarrage
    RATED_KMH   = 50     # vitesse à laquelle on atteint la puissance nominale
    CUT_OUT_KMH = 110    # vitesse de coupure de sécurité

    v = wind_speed_kmh.copy()

    fraction = np.where(
        v < CUT_IN_KMH,
            0.0,
        np.where(
            v < RATED_KMH, 
                ((v - CUT_IN_KMH) / (RATED_KMH - CUT_IN_KMH))**3, # zone cubique
            np.where(
                v < CUT_OUT_KMH, 
                    1.0, #puissance nominale
                    0.0 #arret au dessus cut out
            )
        )
    )

    power = pd.Series(
        np.clip(fraction * WIND_CAPACITY_MW, 0.0, WIND_CAPACITY_MW), 
        index = wind_speed_kmh.index, 
        name="wind_power_mwh"
    )

    return power




def compute_solar_power(radiation_wm2 : pd.Series) -> pd.Series:
    """
    Conversion irradiation solaire → puissance produite (MW).

    Formule :
      P (MW) = (radiation / STC_RADIATION) × SOLAR_CAPACITY_MW

    La logique : à STC_RADIATION = 1000 W/m² (conditions standard),
    les panneaux tournent à pleine capacité. En dessous, la puissance
    est proportionnelle au ratio radiation / 1000.

    Le rendement PANEL_EFFICIENCY est déjà intégré dans capacity_max
    (qui représente la puissance AC en sortie onduleur, pas la puissance
    crête brute des cellules). On l'applique ici comme facteur correctif
    pour tenir compte des pertes thermiques, câblage, et poussière.

    Exemple concret avec SOLAR_CAPACITY_MW = 150 MW :
      radiation =    0 W/m² → P = 0.00 MW  (nuit)
      radiation =  200 W/m² → P = 150 × (200/1000) × 0.18/0.18 = 30 MW
      radiation =  500 W/m² → P = 150 × 0.5 = 75 MW
      radiation = 1000 W/m² → P = 150 MW   (pleine capacité, STC)
      radiation = 1100 W/m² → P = 150 MW   (borné à capacity_max)
      radiation =   -5 W/m² → P = 0 MW     (borné à 0, valeurs nocturnes parasites)

    Note : direct_radiation peut être légèrement négatif la nuit
    (artefact du modèle) → clip à 0 en borne basse.
    """
    SOLAR_CAPACITY_MW = ASSETS["solar"]["capacity_max"]
    STC_RADIATION = 1000    # W/m² — irradiation en conditions standard
    PANEL_EFFICIENCY = 0.18 # rendement typique panneau monocristallin

    power = (radiation_wm2 / STC_RADIATION) * SOLAR_CAPACITY_MW * PANEL_EFFICIENCY / 0.18
    power = power.clip(lower = 0.0, upper = SOLAR_CAPACITY_MW)
    power.name = "solar_power_mw"

    return power


# ─────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────

def fetch_weather_power(
        target_date = TARGET_DATE,
        location = LOCATION,
) -> pd.DataFrame :
    
    """
    Pipeline complet : appel API → parsing → conversion en MW.

    Retourne un DataFrame de 96 lignes (pas 15min) avec :
    - slot            : 0..95
    - datetime        : timestamp Europe/Paris
    - wind_speed_10m  : km/h
    - wind_speed_80m  : km/h
    - solar_radiation : W/m²
    - temperature     : °C
    - wind_power_mw   : MW  ← borne sup éolien pour l'optimiseur
    - solar_power_mw  : MW  ← borne sup solaire pour l'optimiseur
    """

    response = fetch_raw_weather(target_date, location)
    df = parse_weather(response, target_date)

    df["wind_power_mw"]  = compute_wind_power(df["wind_speed_80m"])
    df["solar_power_mw"] = compute_solar_power(df["direct_radiation"])
    

    print("\n[Weather Power] Production estimée sur la journée :")
    print(f"  Éolien  : {df['wind_power_mw'].mean():.1f} MW moy  |  "
          f"{df['wind_power_mw'].sum() * 0.25:.0f} MWh")
    print(f"  Solaire : {df['solar_power_mw'].mean():.1f} MW moy  |  "
          f"{df['solar_power_mw'].sum() * 0.25:.0f} MWh")
    # × 0.25 car chaque slot = 15min = 0.25h

    return df

# ─────────────────────────────────────────────
# EXPORT CSV
# ─────────────────────────────────────────────

def save_weather(df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    """Sauvegarde le DataFrame météo + puissance dans data/raw/."""

    os.makedirs(output_dir, exist_ok=True)
    filename = f"weather_power_{TARGET_DATE}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[SAVE] Météo sauvegardée → {filepath}")
    return filepath

# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_weather_power()
    print("\nAperçu des données météo + puissance :")
    print(df[["slot", "date", "wind_speed_80m",
              "direct_radiation", "wind_power_mw", "solar_power_mw"]].to_string())
    save_weather(df)