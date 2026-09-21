# config.py

from datetime import date

''' ─────────────────────────────────────────────
    MARCHÉ & PÉRIODE
   ─────────────────────────────────────────────'''

PRICE_ZONE = "10YFR-RTE------C"   # Zone de prix ENTSO-E : France (RTE)
TARGET_DATE = date(2026, 9, 13)
CURRENCY = "EUR"
ENERGY_UNIT = "MWh"

# ─────────────────────────────────────────────
# COORDONNÉES GÉOGRAPHIQUES (pour Open-Meteo)
# ─────────────────────────────────────────────

LOCATION = {
    "name": "Paris, FR",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "timezone" : "Europe/Paris"
}

# ─────────────────────────────────────────────
# PORTEFEUILLE D'ACTIFS DE PRODUCTION
# ─────────────────────────────────────────────
# Paramètres physiques classiques par technologie
#
# capacity_min  : puissance minimale technique en MW (must-run)
# capacity_max  : puissance installée maximale en MW
# ramp_up       : rampe montante maximale en MW/h
# ramp_down     : rampe descendante maximale en MW/h
#primary_reserve_timing" : temps de déclenchement de la réserve primaire
#secondary_reserve_timing" : 
#tertiary_reserve_timing" : 
#primary_reserve_puissance" : puissance de la reserve primaire
#secondary_reserve_puissance" :
#tertiary_reserve_puissance" : 
# marginal_cost : coût marginal de production en €/MWh
#                 (pour gaz : calculé dynamiquement dans preprocess.py)
# startup_cost  : coût de démarrage en € (une seule fois si on démarre)
# co2_intensity : émissions en tCO2/MWh (pour coût CO2 si EUA intégré)
# dispatchable  : True = contrôlable, False = fatal (dépend de la météo)
# ─────────────────────────────────────────────

ASSETS = {

    "nuclear": {
        "capacity_min" : 700, # MW  — must-run élevé (contrainte technique)
        "capacity_max" : 1500, # MW
        "ramp_up" : 50, #MWh faible : réacteur peu flexible 
        "ramp_down" : 50, 
        "primary_reserve_timing" : 30, # S : temps de déclenchement de la réserve primaire
        "secondary_reserve_timing" : 15*60, #: 15 minutes
        "tertiary_reserve_timing" : 2*60*60, # 2h : arbitraire
        "primary_reserve_puissance" : 3/100, # puissance disponible
        "secondary_reserve_puissance" : 7/100, #: 15 minutes
        "tertiary_reserve_puissance" : 20/100, # 2h : arbitraire
        "marginal_cost" : 12.00, # €/MWh — très bas
        "startup_cost" : 50_000, # €/MWh coût de démarrage par MWh
        "co2_intensity" : 0.006, # tCO2/MWh
        "dispatchable" : True # modulable 
    },

    "gas_ccgt": {
        "capacity_min" : 0, # MW, peut s'arrêter
        "capacity_max" : 500, # MW
        "ramp_up" : 200, #MWh élevé : très flexible
        "ramp_down" : 200, 
        "primary_reserve_timing" : 30, # S : temps de déclenchement de la réserve primaire
        "secondary_reserve_timing" : 15*60, #: 15 minutes
        "tertiary_reserve_timing" : 2*60*60, # 2h : arbitraire
        "primary_reserve_puissance" : 7/100, # puissance disponible
        "secondary_reserve_puissance" : 35/100, #: 15 minutes
        "tertiary_reserve_puissance" : 1, # 2h : arbitraire
        "marginal_cost" : None, # €/MWh — très bas
        "startup_cost" : 8_000, # €coût de démarrage 
        "co2_intensity" : 0.37, # tCO2/MWh elevé
        "dispatchable" : True # modulable 
    }, 

    "hydro_reservoir" : {
        "capacity_min": 0,         # MW
        "capacity_max": 300,       # MW
        "ramp_up": 300,            # MW/h — quasi instantané
        "ramp_down": 300,          # MW/h
        "marginal_cost": 5.0,      # €/MWh — coût d'opportunité de l'eau
        "startup_cost": 500,       # €
        "co2_intensity": 0.004,    # tCO2/MWh
        "dispatchable": True,
        "daily_energy_budget": 1_800,  # MWh/jour — contrainte de réservoir    
    }, 

    "wind" : {
        "capacity_min": 0,         # MW — fatal : ne peut pas être forcé
        "capacity_max": 200,       # MW — puissance installée
        "ramp_up": 200,            # MW/h — non contraignant (c'est le vent qui contrôle)
        "ramp_down": 200,          # MW/h
        "marginal_cost": 0.0,      # €/MWh — coût marginal nul (pas de combustible)
        "startup_cost": 0,         # €
        "co2_intensity": 0.011,    # tCO2/MWh
        "dispatchable": False,     # production bornée par fetch_weather.py    
    }, 

    "solar": {
        "capacity_min": 0,         # MW
        "capacity_max": 150,       # MW — puissance installée
        "ramp_up": 150,            # MW/h
        "ramp_down": 150,          # MW/h
        "marginal_cost": 0.0,      # €/MWh
        "startup_cost": 0,         # €
        "co2_intensity": 0.048,    # tCO2/MWh
        "dispatchable": False,     # production bornée par irradiation solaire
    },

    "battery": {
        "capacity_min": -100,      # MW — négatif = charge (consommation)
        "capacity_max": 100,       # MW — positif = décharge (production)
        "ramp_up": 100,            # MW/h — instantané
        "ramp_down": 100,          # MW/h
        "marginal_cost": 2.0,      # €/MWh — coût de dégradation cycles
        "startup_cost": 0,         # €
        "co2_intensity": 0.0,      # tCO2/MWh
        "dispatchable": True,
        "capacity_mwh": 200,       # MWh — énergie stockable max
        "efficiency": 0.92,        # rendement aller-retour (charge → décharge)
        "soc_min": 0.10,           # state of charge minimum (10%)
        "soc_max": 0.90,           # state of charge maximum (90%)
        "soc_initial": 0.50,       # état de charge au début de la journée
    }, 
}

# ─────────────────────────────────────────────
# PARAMÈTRES CO2
# ─────────────────────────────────────────────

EUA_PRICE = 65.0    # €/tCO2 — prix du quota carbone européen (EUA)
                    # intégré dans le coût marginal effectif du gaz

# ─────────────────────────────────────────────
# PARAMÈTRES GAZ (pour calcul dynamique du coût CCGT)
# ─────────────────────────────────────────────

GAS_EFFICIENCY_CCGT = 0.58    # rendement thermique du cycle combiné (58%)
# coût marginal gaz (€/MWh_elec) = prix_TTF (€/MWh_gaz) / rendement
# + coût CO2 = co2_intensity × EUA_PRICE
# → calculé dans preprocess.py à partir du prix TTF du jour

# ─────────────────────────────────────────────
# PARAMÈTRES SOLVER
# ─────────────────────────────────────────────

SOLVER = 'PULP_CBC_CMD' #"CBC"          # solver open-source livré avec PuLP (CBC de COIN-OR)
TIME_HORIZON = 24       # heures (journée complète Day-Ahead)
HOURS = list(range(TIME_HORIZON))