import pandas as pd 
from dotenv import load_dotenv
from pathlib import Path
from entsoe import EntsoePandasClient
from datetime import timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRICE_ZONE, TARGET_DATE, LOCATION, HOURS

env_path = Path(__file__).parent / "logs.env"
load_dotenv(env_path)
print(f"Path to .env : {env_path}")


''' ─────────────────────────────────────────────'''
# AUTHENTIFICATION
# ─────────────────────────────────────────────

def get_client() -> EntsoePandasClient:
    api_key =  os.getenv("ENTSOE_API_KEY") # à mettre en place
    print(api_key[0:5])
    if not api_key:
        raise EnvironmentError(
            "Token manquant"
        )
    return EntsoePandasClient(api_key=api_key)

# ─────────────────────────────────────────────
# EXTRACTION DES PRIX DAY-AHEAD
# ─────────────────────────────────────────────
def fetch_day_ahead_prices(
        target_date=TARGET_DATE,
        price_zone = PRICE_ZONE,
) -> pd.Series:
    """
    Récupère les prix Day-Ahead horaires (€/MWh) depuis ENTSO-E
    pour une zone de prix et une date données.

    Retourne une pd.Series de 24*4 valeurs indexées par heure (0 à 23),
    avec le prix en €/MWh pour chaque heure de la journée.

    Paramètres
    ----------
    target_date : date
        La date pour laquelle on veut les prix DA (ex: date(2024, 1, 15))
    price_zone  : str
        Code EIC de la zone de prix ENTSO-E (ex: "10YFR-RTE------C" pour France)

    Codes EIC utiles
    ----------------
    France          : 10YFR-RTE------C
    Allemagne       : 10Y1001A1001A82H
    Espagne         : 10YES-REE------0
    Italie Nord     : 10Y1001A1001A73I
    Belgique        : 10YBE----------2
    Pays-Bas        : 10YNL----------L
    Grande-Bretagne : 10YGB----------A
    """

    client = get_client()

    #ENTSOE attend des timestamps avec timzone
    #On va couvrir J 00:00 à J+1 00h00 pour avoir les 24h
    start = pd.Timestamp(target_date, tz=LOCATION["timezone"])#"Europe/Paris")#"UTC")
    end = pd.Timestamp(target_date + timedelta(days = 1), tz=LOCATION["timezone"])#"Europe/Paris")#"UTC")

    print(f"[ENTSO-E] Récupération des prix DA — zone : {price_zone} — date : {target_date}")

    try : 
        raw: pd.Series = client.query_day_ahead_prices(
            country_code = price_zone,
            start = start, 
            end = end
        )
    except Exception as e :
        raise RuntimeError(f"Erreur lors de l'appel API ENTSOE-E : {e}")
    
    # ── Nettoyage & validation ──────────────────────────────────────────

    # Convertit l'index UTC en heure locale Europe/Paris si besoin
    raw.index = raw.index.tz_convert(LOCATION["timezone"])#"Europe/Paris")

    # Filtre uniquement les heures du jour cible (sécurité DST)
    mask = raw.index.date == target_date
    prices = raw[mask].copy()

    # Vérification du nombre de valeurs
    n = len(prices)
    if n not in (23, 24, 25, 92, 96, 98):
        raise ValueError(
            f"Nombre d'heures inattendu : {n}"
        )
    if (n !=24 and n != 96):
        print(f"[ATTENTION] Changement d'heure détecté : {n} heures pour le {target_date}")

    # Réindexe sur 0..N-1 (heures de la journée) pour simplifier l'optimiseur
    prices.index = range(n)
    prices.name = "da_price_eur_mwh"

    #Check l'absence de NaN
    if prices.isnull().any():
        missing = prices[prices.isnull()].index.tolist()
        raise ValueError(f"Prix manquants pour les heures : {missing}")

    # Vérifie l'absence de prix aberrants (prix négatifs profonds ou spike extrême)
    _validate_prices(prices)

    print(f"[ENTSO-E] OK — {n} heures récupérées")
    print(f"Prix min : {prices.min():.2f} €/MWh   |  "
          f"Prix max : {prices.max():.2f} €/MWh   |  "
          f"Prix moyen : {prices.mean():.2f} €/MWh   |  ")
    
    return prices

''' ─────────────────────────────────────────────'''
# VALIDATION
# ─────────────────────────────────────────────

def _validate_prices(prices : pd.Series) -> None:
    """
    Contrôles de cohérence sur les prix récupérés.
    Lève des warnings sans bloquer si les seuils sont dépassés.
    """

    PRICE_MIN_WARN  = -100    # €/MWh — prix négatifs de + en plus courants
    PRICE_MIN_ERROR = -500    # €/MWh — en dessous : probablement une erreur
    PRICE_MAX_WARN  =  500    # €/MWh — spike élevé mais possible 
    PRICE_MAX_ERROR = 3_000   # €/MWh — au dessus : probablement une erreur

    if prices.min() < PRICE_MIN_ERROR:
        raise ValueError(f"Prix anormalement bas : {prices.min():.2f} €/MWh")
    if prices.max() > PRICE_MAX_ERROR:
        raise ValueError(f"Prix anormalement haut : {prices.max():.2f} €/MWh")
    if prices.min() < PRICE_MIN_WARN:
        print(f"[WARN] Prix négatifs détectés (min : {prices.min():.2f} €/MWh) "
              f"— phénomène courant lors de surplus renouvelable")
    if prices.max() > PRICE_MAX_WARN:
        print(f"[WARN] Spike de prix détecté (max : {prices.max():.2f} €/MWh)")

# ─────────────────────────────────────────────
# EXPORT CSV 
# ─────────────────────────────────────────────

def save_prices(
        prices: pd.Series,
        output_dir: str = "data/raw", 
        target_date=TARGET_DATE
) -> str:
   """Sauvegarde les prix dans un CSV horodaté."""


   start = pd.Timestamp(target_date, tz="Europe/Paris")#"UTC")
   end = pd.Timestamp(target_date + timedelta(days = 1), tz="Europe/Paris")#"UTC")

   datetime_index = pd.date_range(
       start=start,
       end=end,
       freq="15min" if len(prices) in (92, 96, 100) else "h",
       inclusive="left"  # exclut le end
   )

   if len(datetime_index) != len(prices):
       raise ValueError(
           "Lenghts différentes :" 
           f"{len(datetime_index)} timestamps vs " 
           f"{len(prices)} prix"
       )
   
   df = pd.DataFrame({
       "date" : datetime_index,
       "da_price_eur_mwh" : prices.values
   })
   filename_dates = f"da_prices_{prices.name}_{TARGET_DATE}_dates.csv"
   filepath_dates = os.path.join(output_dir, filename_dates)

   os.makedirs(output_dir, exist_ok = True)
   filename = f"da_prices_{prices.name}_{TARGET_DATE}.csv"
   filepath = os.path.join(output_dir, filename)

   prices.to_csv(filepath, header = True)
   df.to_csv(filepath_dates, header = True)
   print(f"[SAVE] Prix sauvegardés -> {filepath}")
   return filepath

# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    prices = fetch_day_ahead_prices()
    print("\nPrix Day-Ahead récupérés :")
    print(prices.to_string())
    save_prices(prices)