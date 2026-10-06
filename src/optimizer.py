"""
optimizer.py — Modèle de Programmation Linéaire (LP) pour le dispatch
Day-Ahead d'un portefeuille de production d'énergie.

Problème résolu :
  Maximiser le profit journalier d'un producteur d'énergie en décidant,
  pour chaque slot de 15min, la puissance produite par chaque actif.

Formulation mathématique :
  ┌─────────────────────────────────────────────────────────────────────┐
  │  MAX  Σ_t Σ_a [ price(t) × P(a,t) - marginal_cost(a) × P(a,t) ]   │
  │       × SLOT_DURATION_H                                             │
  │                                                                     │
  │  sous contraintes :                                                 │
  │    (1) pmin(a) ≤ P(a,t) ≤ pmax(a,t)    capacité                   │
  │    (2) |P(a,t) - P(a,t-1)| ≤ ramp(a)   flexibilité                │
  │    (3) Σ_t P(hydro,t) × Δt ≤ budget     réservoir                  │
  │    (4) SOC(t) = SOC(t-1) + charge/décharge  batterie               │
  │    (5) SOC_min ≤ SOC(t) ≤ SOC_max       état de charge             │
  └─────────────────────────────────────────────────────────────────────┘

  où :
    t ∈ {0..95}       slots de 15min sur une journée
    a ∈ {nuclear, gas, hydro, wind, solar, battery}   actifs
    P(a,t)            variable de décision : puissance en MW
    SLOT_DURATION_H   = 0.25h (conversion MW → MWh)

Librairie utilisée : PuLP (wrapper Python autour du solver CBC de COIN-OR)
  → CBC est un solver open-source de qualité industrielle pour les LP/MIP
  → Gratuit, aucune licence nécessaire, livré avec PuLP
"""

import pandas as pd
import numpy as np
import pulp
import os 
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_DATE, ASSETS, SOLVER

'''
# ─────────────────────────────────────────────
# CONSTANTES 
# ─────────────────────────────────────────────
'''

SLOT_DURATION_H = 0.25 # 15min = 0.25 heure  (MW × 0.25h = MWh)
N_SLOTS = 96 # nombre de pas dans la journée
SLOTS = list(range(N_SLOTS))

ASSETS_DISPATCH = ["nuclear", "gas", "hydro", "wind", "solar", "battery"]
# Ordre = ordre dans la merit order classique (du moins cher au plus cher)
# nuclear (12€) < hydro (5€) < wind (0€) < solar (0€) < battery (2€) < gas (93€)
# Note : hydro et renouvelables passent devant le gaz en merit order réel

'''
# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
'''

def load_optimizer_input(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Charge le CSV produit par preprocess.py.
    Vérifie la présence de toutes les colonnes critiques.
    """
    filepath = os.path.join(data_dir, f"optimizer_input_{TARGET_DATE}.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Fichier optimizer_input introuvable : {filepath}\n"
            "Lance d'abord : python src/preprocess.py"
        )
    
    df = pd.read_csv(filepath)
    # Parse le datetime (perdu à l'export CSV)
    df["datetime"] = pd.to_datetime(df["datetime"], utc = True).dt.tz_convert("Europe/Paris")

    print(f"[LOAD] optimizer_input — {len(df)} slots × {len(df.columns)} colonnes")
    return df

'''
# ─────────────────────────────────────────────
# 2. CONSTRUCTION DU MODÈLE LP
# ─────────────────────────────────────────────
'''

def build_lp_model(df: pd.DataFrame) -> tuple:
    """
    Construit le problème LP PuLP complet.

    Retourne : (prob, P, SOC)
      prob : objet LpProblem PuLP (le modèle)
      P    : dict[asset][slot] → variable de décision puissance (MW)
      SOC  : dict[slot] → variable état de charge batterie (MWh)

    Structure PuLP :
      LpVariable      = variable de décision (ce que le solver optimise)
      LpProblem       = conteneur du modèle (objectif + contraintes)
      prob += expr    = ajoute une contrainte ou définit l'objectif
    """

    # ── Initialisation du problème ────────────────────────────────────────
    # LpMaximize = on cherche à MAXIMISER la fonction objectif (profit)  
    prob = pulp.LpProblem(
        name = f"day_ahead_dispatch_{TARGET_DATE}",
        sense = pulp.LpMaximize,
    )
    print("[LP] Initialisation du modèle PuLP...")

    # ── Variables de décision : puissance par actif et par slot ───────────
    #
    # P[asset][t] = puissance produite (MW) par l'actif 'asset' au slot t
    #
    # lowBound / upBound : bornes statiques (seront affinées par des
    # contraintes dynamiques pour les renouvelables et la batterie)
    #
    # Pour la batterie : P peut être négatif (charge) ou positif (décharge)
    # → lowBound = capacity_min = -100 MW
    # Pour tous les autres : P ≥ 0

    P = {}  # P[asset][slot] → LpVariable

    for asset in ASSETS_DISPATCH:
        P[asset] = {}
        cfg = ASSETS[asset] if asset in ASSETS else {}

        p_lo = cfg.get("capacity_min", 0)
        p_hi = cfg.get("capacity_max", 0)

        for t in SLOTS:
            P[asset][t] = pulp.LpVariable(
                name=f"P_{asset}_{t}",
                lowBound = p_lo,
                upBound = p_hi,
                cat = "Continuous",
            )

    # ── Variable d'état de charge batterie SOC(t) ─────────────────────────
    #
    # SOC[t] = énergie stockée dans la batterie au début du slot t (MWh)
    # Bornes : [SOC_min × capacity_mwh, SOC_max × capacity_mwh]
    # Ex : capacity=200 MWh, soc_min=0.10, soc_max=0.90
    #   → SOC ∈ [20 MWh, 180 MWh]

    bat = ASSETS["battery"]
    SOC_MIN_MWH = bat["soc_min"] * bat["capacity_mwh"]
    SOC_MAX_MWH = bat["soc_max"] * bat["capacity_mwh"]
    SOC_INIT = bat["soc_initial"] * bat["capacity_mwh"]

    SOC = {}
    for t in SLOTS:
        SOC[t] = pulp.LpVariable(
            name = f"SOC_{t}",
            lowBound = SOC_MIN_MWH,
            upBound = SOC_MAX_MWH,
            cat = "Continuous",
        )

    print(f"[LP] Variables créées : "
          f"{len(ASSETS_DISPATCH) * N_SLOTS} puissance + {N_SLOTS} SOC "
          f"= {len(ASSETS_DISPATCH) * N_SLOTS + N_SLOTS} variables au total")  
    
        # ── Fonction objectif ────────────────────────────────────────────────
    #
    # Profit = Σ_t Σ_a [ (prix_DA(t) - coût_marginal(a)) × P(a,t) × Δt ]
    #
    # On maximise la marge brute (revenue - fuel cost - CO2 cost).
    # Les coûts fixes (amortissement, maintenance) ne sont pas modélisés
    # car ils ne dépendent pas des décisions de dispatch (ils sont "sunk").
    #
    # Note sur la batterie : son "coût marginal" représente la dégradation
    # des cellules. On l'applique sur la valeur absolue de P[battery][t]
    # (charge ET décharge usent la batterie). Mais PuLP est linéaire :
    # on ne peut pas faire abs(). Astuce : on pénalise la décharge (P > 0)
    # avec le marginal_cost, et la charge (P < 0) génère un gain car on
    # achète de l'énergie bon marché pour la revendre plus tard.
    # La dégradation réelle serait modélisée en MIP (hors scope ici).

    marginal_costs = {
        "nuclear"  : df["nuclear_marginal_cost"].iloc[0],   # 12.0 €/MWh
        "gas"      : df["gas_marginal_cost"].iloc[0],        # ~93.0 €/MWh
        "hydro"    : df["hydro_marginal_cost"].iloc[0],      # 5.0 €/MWh
        "wind"     : 0.0,
        "solar"    : 0.0,
        "battery"  : df["battery_marginal_cost"].iloc[0],    # 2.0 €/MWh
    }

    # Construction de la somme de profit sur tous les slots et actifs
    # PuLP supporte lpSum() pour les grandes sommes → plus efficace que +=
    profit_terms = []
    for t in SLOTS:
        price_t = df["da_price_eur_mwh"].iloc[t]   # €/MWh au slot
        for asset in ASSETS_DISPATCH:
            margin = price_t - marginal_costs[asset] # marge unitaire €/MWh
            profit_terms.append(
                margin * P[asset][t] * SLOT_DURATION_H # €/MWh
            )

    # L'objectif est la somme de toutes ces marges
    prob += pulp.lpSum(profit_terms), "Profit_total_EUR"

    print(f"[LP] Fonction objectif définie ({len(profit_terms)} termes)")   

    # ─────────────────────────────────────────────────────────────────────
    # 3. CONTRAINTES
    # ─────────────────────────────────────────────────────────────────────
    n_constraints = 0

    # ── C1 : Bornes dynamiques des renouvelables ──────────────────────────
    #
    # Les variables P[wind][t] et P[solar][t] ont déjà des bornes statiques
    # définies à la création (0, capacity_max). Mais leur borne MAX réelle
    # dépend de la météo heure par heure (wind_pmax, solar_pmax).
    # On ajoute des contraintes explicites slot par slot.
    #
    # P[wind][t] ≤ wind_pmax(t)    pour tout t
    # P[solar][t] ≤ solar_pmax(t)  pour tout t

    for t in SLOTS:
        wind_available  = df["wind_pmax"].iloc[t]   # MW disponible ce slot
        solar_available = df["solar_pmax"].iloc[t]  # MW disponible ce slot

        prob += (
            P["wind"][t] <= wind_available,
            f"Wind_pmax_{t}",
        )
        prob += (
            P["solar"][t] <= solar_available,
            f"Solar_pmax_{t}",
        )
        n_constraints += 2

    print(f"[LP] C1 Bornes renouvelables : {n_constraints} contraintes")  

    # ── C2 : Contraintes de rampe ─────────────────────────────────────────
    #
    # Un actif thermique ne peut pas changer de puissance instantanément.
    # Entre deux slots consécutifs, la variation est limitée :
    #
    #   P(a,t) - P(a,t-1) ≤  ramp_up(a)      (montée limitée)
    #   P(a,t-1) - P(a,t) ≤  ramp_down(a)    (descente limitée)
    #
    # Les rampes sont en MW/slot (déjà converties dans preprocess.py × 0.25)
    # Ex : nucléaire ramp_up = 50 MW/h × 0.25h = 12.5 MW/slot

    ramp_assets = ["nuclear", "gas", "hydro", "battery"]
    # Renouvelables exclus : le vent/soleil n'ont pas de contrainte de rampe
    # (on peut "curtailer" instantanément)

    c2_count = 0
    for asset in ramp_assets:
        ramp_up_col   = f"{asset}_ramp_up"    # colonne dans df (MW/slot)
        ramp_down_col = f"{asset}_ramp_down"

        # La rampe est constante sur la journée → on prend iloc[0]
        ramp_up   = df[ramp_up_col].iloc[0]    # MW/slot
        ramp_down = df[ramp_down_col].iloc[0]  # MW/slot

        for t in SLOTS[1:]:    # commence à t=1 (pas de t-1 pour t=0)
            # Contrainte montée : P(t) - P(t-1) ≤ ramp_up
            prob += (
                P[asset][t] - P[asset][t-1] <= ramp_up,
                f"Ramp_up_{asset}_{t}",
            )
            # Contrainte descente : P(t-1) - P(t) ≤ ramp_down
            prob += (
                P[asset][t-1] - P[asset][t] <= ramp_down,
                f"Ramp_down_{asset}_{t}",
            )
            c2_count += 2

    n_constraints += c2_count
    print(f"[LP] C2 Rampes : {c2_count} contraintes")

    # ── C3 : Budget énergétique journalier hydraulique ────────────────────
    #
    # Le réservoir contient une quantité d'eau finie.
    # On ne peut pas produire plus que le budget journalier, quelle que
    # soit la puissance instantanée.
    #
    #   Σ_t P(hydro,t) × Δt ≤ daily_energy_budget   (MWh)
    #
    # Intuition : si tu produis toujours à 300 MW pendant 96 slots de 15min :
    #   300 MW × 96 × 0.25h = 7200 MWh >> budget de 1800 MWh
    # → la contrainte force l'hydro à n'être utilisé qu'aux heures de pic.
    # C'est son rôle économique : réserve flexible pour les pointes.

    '''
    C3a : Bilan volumique réservoir amont (slot par slot)
      V_res(t+1) = V_res(t) - Q_turb(t) + Q_apport(t) - Q_debit_reserve
      → le volume évolue à chaque slot selon turbinage, apports, débit réservé

    C3b : Niveaux min/max réservoir amont
        level_env_min ≤ V_res(t) ≤ level_max    pour tout t

    C3c : Bilan volumique bassin aval avec délai d'écoulement
        V_aval(t+1) = V_aval(t) + Q_turb(t - delay) - Q_pompage(t)
        → l'eau turbinée arrive dans le bassin aval avec un délai

    C3d : Niveaux min/max bassin aval
        level_env_min_aval ≤ V_aval(t) ≤ level_max_aval

    C3e : Lien puissance turbinée ↔ volume turbiné
        Q_turb(t) = P_hydro(t) × SLOT_DURATION_H / mwh_per_hm3

    C3f : STEP — bilan de pompage
        Q_pompe(t) = P_step_pump(t) × SLOT_DURATION_H × hm3_per_mwh_pumped

    C3g : Débit réservé minimum (environnemental)
        Q_turb(t) ≥ min_flow_hm3_per_slot    pour tout t

    C3h : Niveau cible fin de journée (optionnel)
        V_res(95) ≥ level_target_hm3
    '''
    '''
    hydro_budget = ASSETS["hydro_reservoir"]["daily_energy_budget"]

    prob += (
        pulp.lpSum(P["hydro"][t] * SLOT_DURATION_H for t in SLOTS) <= hydro_budget,
        "Hydro_daily_budget",
    )
    n_constraints += 1
    print(f"[LP] C3 Budget hydro : 1 contrainte (budget={hydro_budget} MWh/j)")
    '''

    # ── Variables hydrauliques avancées ──────────────────────────────────
    from config import HYDRO

    # Volume réservoir amont (Hm³) — état du lac slot par slot
    V_res = {}
    for t in SLOTS:
        V_res[t] = pulp.LpVariable(
            name     = f"V_res_{t}",
            lowBound = HYDRO["reservoir"]["level_env_min_hm3"],  # 10 Hm³
            upBound  = HYDRO["reservoir"]["level_max_hm3"],      # 47.5 Hm³
            cat      = "Continuous",
        )

    # Volume bassin aval (Hm³)
    V_aval = {}
    for t in SLOTS:
        V_aval[t] = pulp.LpVariable(
            name     = f"V_aval_{t}",
            lowBound = HYDRO["lower_basin"]["level_env_min_hm3"],  # 2 Hm³
            upBound  = HYDRO["lower_basin"]["level_max_hm3"],      # 9.5 Hm³
            cat      = "Continuous",
        )

    # Puissance de pompage STEP (MW) — toujours positive
    # La STEP consomme quand elle pompe → comptée comme charge dans le bilan
    P_pump = {}
    for t in SLOTS:
        P_pump[t] = pulp.LpVariable(
            name     = f"P_pump_{t}",
            lowBound = 0,
            upBound  = HYDRO["step"]["pump_capacity_mw"],   # 100 MW
            cat      = "Continuous",
        )


    # ── C3 : Modélisation hydraulique avancée ─────────────────────────────
    from config import HYDRO

    mwh_per_hm3      = HYDRO["mwh_per_hm3"]           # 245 MWh/Hm³
    min_flow         = HYDRO["min_flow_hm3_per_slot"]  # 0.002 Hm³/slot
    flow_delay       = HYDRO["flow_delay_slots"]       # 2 slots
    pump_eff         = HYDRO["step"]["pump_efficiency"]        # 0.88
    hm3_per_mwh_pump = HYDRO["step"]["hm3_per_mwh_pumped"]    # 0.0046

    c3_count = 0

    # ── C3a : Condition initiale réservoir amont ──────────────────────────
    prob += (
        V_res[0] == HYDRO["reservoir"]["level_initial_hm3"],
        "Hydro_res_init",
    )
    c3_count += 1

    # ── C3b : Condition initiale bassin aval ──────────────────────────────
    prob += (
        V_aval[0] == HYDRO["lower_basin"]["level_initial_hm3"],
        "Hydro_aval_init",
    )
    c3_count += 1

    # ── C3c/C3d : Bilan volumique slot par slot ───────────────────────────
    #
    # Réservoir amont :
    #   V_res(t+1) = V_res(t)
    #              - Q_turb(t)          [eau turbinée vers l'aval]
    #              + Q_apport(t)        [pluie → ruissellement]
    #              - min_flow           [débit réservé obligatoire]
    #              + Q_pompe(t)         [eau remontée par la STEP]
    #
    # où Q_turb(t) = P_hydro(t) × SLOT_DURATION_H / mwh_per_hm3
    #    Q_pompe(t) = P_pump(t) × SLOT_DURATION_H × hm3_per_mwh_pump

    for t in SLOTS[:-1]:
        q_turb_t   = P["hydro"][t] * SLOT_DURATION_H / mwh_per_hm3
        q_apport_t = df["hydro_inflow_hm3"].iloc[t]
        q_pompe_t  = P_pump[t] * SLOT_DURATION_H * hm3_per_mwh_pump

        # Bilan réservoir amont
        prob += (
            V_res[t+1] == V_res[t]
                        - q_turb_t
                        + q_apport_t
                        - min_flow
                        + q_pompe_t,
            f"Hydro_res_balance_{t}",
        )

        # Bilan bassin aval avec délai d'écoulement
        # L'eau turbinée au slot t arrive dans l'aval au slot t + flow_delay
        # Pour les premiers slots où t - flow_delay < 0 → pas d'arrivée
        t_source = t - flow_delay
        if t_source >= 0:
            q_arrive_t = P["hydro"][t_source] * SLOT_DURATION_H / mwh_per_hm3
        else:
            q_arrive_t = 0.0   # pas encore arrivé au début de journée

        prob += (
            V_aval[t+1] == V_aval[t]
                         + q_arrive_t    # eau arrivant de l'amont
                         - q_pompe_t,   # eau remontée par la STEP
            f"Hydro_aval_balance_{t}",
        )
        c3_count += 2

    # ── C3e : Débit réservé minimum (contrainte environnementale) ─────────
    #
    # À chaque slot, le turbinage doit être suffisant pour garantir
    # le débit réservé dans la rivière (obligation légale).
    # Q_turb(t) × mwh_per_hm3 / SLOT_DURATION_H ≥ min_flow
    # → P_hydro(t) ≥ min_flow × mwh_per_hm3 / SLOT_DURATION_H

    min_power_env = min_flow * mwh_per_hm3 / SLOT_DURATION_H  # MW

    for t in SLOTS:
        prob += (
            P["hydro"][t] >= min_power_env,
            f"Hydro_env_flow_{t}",
        )
        c3_count += 1

    # ── C3f : Niveau cible fin de journée (optionnel) ─────────────────────
    #
    # Évite de vider complètement le réservoir en fin de journée.
    # Le gestionnaire du barrage doit planifier pour le lendemain.
    level_target = HYDRO["reservoir"].get("level_target_hm3")
    if level_target is not None:
        prob += (
            V_res[N_SLOTS - 1] >= level_target,
            "Hydro_res_target_eod",
        )
        c3_count += 1

    # ── C3g : Anti-simultanéité turbinage / pompage ───────────────────────
    #
    # On ne peut pas turbiner et pomper en même temps sur le même ouvrage.
    # En LP pur (sans variable binaire), on ajoute une contrainte de somme :
    #   P_hydro(t) × SLOT_DURATION_H / mwh_per_hm3 + P_pump(t) ≤ max(turb, pump)
    #
    # C'est une approximation — la vraie contrainte nécessiterait du MIP.
    # En pratique, le prix de l'électricité guide naturellement l'arbitrage :
    # prix haut → turbinage, prix bas → pompage.

    max_combined = max(
        HYDRO["step"]["turb_capacity_mw"],
        HYDRO["step"]["pump_capacity_mw"],
    )
    for t in SLOTS:
        prob += (
            P["hydro"][t] + P_pump[t] <= max_combined,
            f"Hydro_no_simultaneous_{t}",
        )
        c3_count += 1

    n_constraints += c3_count
    print(f"[LP] C3 Hydro avancé : {c3_count} contraintes")
    print(f"     Débit réservé    : {min_flow:.4f} Hm³/slot "
          f"→ {min_power_env:.1f} MW min")
    print(f"     Délai écoulement : {flow_delay} slots ({flow_delay*15} min)")
    print(f"     Pompage STEP max : {HYDRO['step']['pump_capacity_mw']} MW")
    # ── C4 : Dynamique de la batterie (bilan d'énergie) ──────────────────
    #
    # L'état de charge évolue slot après slot selon :
    #
    #   Si P[battery][t] ≥ 0 (décharge) :
    #     SOC(t+1) = SOC(t) - P(t) × Δt / efficiency
    #     → on tire plus d'énergie de la batterie que ce qu'on injecte au réseau
    #       (pertes de rendement à la décharge)
    #
    #   Si P[battery][t] < 0 (charge) :
    #     SOC(t+1) = SOC(t) + |P(t)| × Δt × efficiency
    #     → on stocke moins d'énergie que ce qu'on tire du réseau
    #       (pertes de rendement à la charge)
    #
    # Problème : la formule dépend du SIGNE de P[battery][t], ce qui est
    # non-linéaire. En LP pur, on utilise l'approximation suivante :
    #
    #   SOC(t+1) = SOC(t) - P[battery][t] × Δt × sqrt(efficiency)
    #
    # Justification : si efficiency = 0.92, sqrt(0.92) ≈ 0.959
    # On applique la même pénalité à la charge et à la décharge,
    # ce qui est une approximation conservative mais linéaire.
    # Une modélisation exacte nécessiterait des variables binaires (MIP).

    eff     = bat["efficiency"] # 0.92
    eff_rt  = eff ** 0.5 # rendement "par sens" ≈ 0.959

    # Condition initiale : SOC au slot 0 = SOC_INIT
    prob += (SOC[0] == SOC_INIT, "Battery_SOC_init")
    n_constraints += 1

    # Bilan d'énergie pour chaque transition t → t+1
    for t in SLOTS[:-1]:    # de t=0 à t=94 (t+1 va jusqu'à 95)
        prob += (
            SOC[t+1] == SOC[t] - P["battery"][t] * SLOT_DURATION_H * eff_rt,
            f"Battery_SOC_balance_{t}",
        )
        n_constraints += 1

    # Condition terminale optionnelle : on peut exiger que la batterie
    # finisse avec au moins son SOC initial (évite de "vider" la batterie
    # en fin de journée sans plan de recharge pour le lendemain)
    prob += (SOC[N_SLOTS - 1] >= SOC_INIT, "Battery_SOC_terminal")
    n_constraints += 1

    print(f"[LP] C4 Batterie SOC : {N_SLOTS + 1} contraintes")

    # ── C5 : Nucléaire must-run ───────────────────────────────────────────
    #
    # Le nucléaire a une contrainte de puissance minimale technique.
    # Un réacteur ne peut pas descendre en dessous de ~50% de sa puissance
    # nominale sans risque physique. On a défini pmin = 700 MW dans config.
    #
    # Cette contrainte est déjà encodée dans la borne lowBound de la variable
    # P[nuclear][t] = 700 MW. Pas besoin de la ré-ajouter explicitement.
    #
    # MAIS : on peut vouloir vérifier que le nucléaire ne DÉPASSE pas
    # sa rampe même depuis la condition initiale (slot 0).
    # On suppose qu'au slot 0, le nucléaire tourne déjà à capacity_min.

    # Condition initiale implicite : P[nuclear][0] ≥ pmin (déjà dans lowBound)
    # Pas de contrainte supplémentaire nécessaire ici.^

    # ── C6 : Contrainte de demande (part de marché) ───────────────────────
    #
    # Le portefeuille ne peut produire que sa part de la demande totale.
    # Sans cette contrainte, l'optimiseur produit au maximum dès que
    # prix_DA > coût_marginal — ce qui est irréaliste.
    #
    # Formulation :
    #   Σ_a P(a,t) ≤ demand(t) × MARKET_SHARE    pour tout t
    #
    # La batterie est incluse dans le lpSum : en charge (P < 0) elle
    # réduit la production nette et relâche naturellement la contrainte.
    #
    # Protection anti-infeasible : la borne ne peut pas descendre sous
    # le must-run nucléaire (700 MW), sinon le LP devient infaisable.

    from config import MARKET_SHARE

    nuclear_must_run = ASSETS["nuclear"]["capacity_min"]   # 700 MW
    c6_count = 0

    for t in SLOTS:
        demand_t  = df["demand_mw"].iloc[t]
        borne_t   = demand_t * MARKET_SHARE

        # Garantit que la borne est toujours ≥ must-run nucléaire + marge
        borne_t   = max(borne_t, nuclear_must_run + 50)

        prob += (
            pulp.lpSum(P[asset][t] for asset in ASSETS_DISPATCH) <= borne_t,
            f"Demand_market_share_{t}",
        )
        c6_count += 1

    n_constraints += c6_count
    print(f"[LP] C6 Demande (share={MARKET_SHARE*100:.0f}%) : "
          f"{c6_count} contraintes  "
          f"(borne moy : {df['demand_mw'].mean() * MARKET_SHARE:,.0f} MW  "
          f"min protégée : {nuclear_must_run + 50:,.0f} MW)")

    print(f"\n[LP] Total contraintes : {n_constraints}")
    print(f"[LP] Modèle prêt : {len(prob.variables())} variables, "
          f"{len(prob.constraints)} contraintes")

    return prob, P, SOC, V_res, V_aval, P_pump

'''
# ─────────────────────────────────────────────
# 4. RÉSOLUTION
# ─────────────────────────────────────────────
'''

def solve(prob: pulp.LpProblem) -> str:
    """
    Lance le solver CBC sur le modèle LP.

    Statuts possibles retournés par PuLP :
      "Optimal"    → solution trouvée et prouvée optimale ✓
      "Infeasible" → les contraintes sont contradictoires (impossible à satisfaire)
      "Unbounded"  → la fonction objectif peut croître sans limite (erreur de modèle)
      "Not Solved" → le solver n'a pas été lancé

    En pratique pour notre modèle :
    - "Infeasible" peut arriver si les rampes + must-run + budget hydro
      créent une contradiction (ex : nuclear_pmin trop haut + ramp trop faible)
    - "Optimal" est le cas normal
    """
    print("\n[SOLVE] Lancement du solver CBC...")
    
    #SOLVER = "PULP_CBC_CMD"
    solver = pulp.getSolver(SOLVER, msg = False) #msg=False = pas de log verbeux
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print(f"[SOLVE] Status : {status}")

    if status != "Optimal":
        raise RuntimeError(
            f"Le solver n'a pas trouvé de solution optimale : {status}\n"
            "Vérifie les contraintes (rampes, must-run, budget hydro)."
        )

    if status == "Infeasible":
        # ── Diagnostic de faisabilité ─────────────────────────────────────
        # Affiche les contraintes qui pourraient être en conflit
        print("\n[DEBUG] Analyse des contraintes potentiellement en conflit :")

        # Regroupe par préfixe pour identifier le groupe problématique
        constraint_groups = {}
        for name in prob.constraints:
            prefix = name.split("_")[0] + "_" + name.split("_")[1] \
                     if "_" in name else name
            constraint_groups[prefix] = constraint_groups.get(prefix, 0) + 1

        print("  Groupes de contraintes dans le modèle :")
        for group, count in sorted(constraint_groups.items()):
            print(f"    {group:<35} : {count} contraintes")

        raise RuntimeError(
            f"Le solver n'a pas trouvé de solution optimale : {status}\n"
        )
    
    print(f"[SOLVE] Profit optimal : {pulp.value(prob.objective):,.2f} €")
    return status

'''
# ─────────────────────────────────────────────
# 5. EXTRACTION DES RÉSULTATS
# ─────────────────────────────────────────────
'''

def extract_results(
    df: pd.DataFrame,
    P: dict,
    SOC: dict,
) -> pd.DataFrame:
    """
    Transforme les variables de décision PuLP en DataFrame lisible.

    Pour chaque slot t, on extrait :
    - La puissance de chaque actif (MW)
    - L'énergie produite par chaque actif (MWh = MW × 0.25h)
    - Le SOC de la batterie (MWh)
    - Le revenu brut (€)
    - Le coût de production (€)
    - La marge nette (€)

    pulp.value(variable) → float : extrait la valeur optimale d'une LpVariable
    """

    results = []
    for t in SLOTS:
        price_t = df["da_price_eur_mwh"].iloc[t]

        row = {
            "slot" : t,
            "datetime" : df["datetime"].iloc[t],
            "da_price" : price_t
        }
        if "demand_mw" in df.columns:
            row["demand_mw"] = df["demand_mw"].iloc[t]

        total_production_mw = 0.0
        total_revenue       = 0.0
        total_fuel_cost     = 0.0     

        for asset in ASSETS_DISPATCH:
            p_val = pulp.value(P[asset][t]) # MW optimal pour cet actif ce slot
            if p_val is None:
                p_val = 0.0

            e_val = p_val * SLOT_DURATION_H #MWh produits ce slot

            #Coût marginal de l'actif
            mc = df.get(f"{asset}_marginal_cost",
                        pd.Series([0.0])).iloc[0] if f"{asset}_marginal_cost" in df.columns else 0.0
            
            row[f"P_{asset}_mw"]  = round(p_val, 3)
            row[f"E_{asset}_mwh"] = round(e_val, 4)

            # Pour la batterie en charge (P < 0) : c'est un coût d'achat
            # Pour tous les actifs en production (P > 0) : c'est un coût de combustible
            if asset != "battery":
                total_production_mw += max(p_val, 0)
                total_revenue       += price_t * e_val          # €
                total_fuel_cost     += mc * abs(e_val)          # €

        # Batterie : traitement séparé
        bat_p = pulp.value(P["battery"][t]) or 0.0
        bat_e = bat_p * SLOT_DURATION_H
        bat_mc = ASSETS["battery"]["marginal_cost"]

        if bat_p > 0:    # décharge → vente d'énergie
            total_revenue   += price_t * bat_e
            total_fuel_cost += bat_mc * bat_e    # coût de dégradation
        else:            # charge → achat d'énergie (coût = prix DA × énergie chargée)
            total_fuel_cost += price_t * abs(bat_e)   # on paye pour charger

        row["soc_battery_mwh"] = round(pulp.value(SOC[t]) or 0.0, 3)
        row["total_prod_mw"]   = round(total_production_mw, 2)
        row["revenue_eur"]     = round(total_revenue, 2)
        row["fuel_cost_eur"]   = round(total_fuel_cost, 2)
        row["margin_eur"]      = round(total_revenue - total_fuel_cost, 2)
        row["V_res_hm3"]    = round(pulp.value(V_res[t])  or 0.0, 4)
        row["V_aval_hm3"]   = round(pulp.value(V_aval[t]) or 0.0, 4)
        row["P_pump_mw"]    = round(pulp.value(P_pump[t])  or 0.0, 3)

        results.append(row)

    results_df = pd.DataFrame(results)

    # Résumé journalier
    total_margin  = results_df["margin_eur"].sum()
    total_revenue = results_df["revenue_eur"].sum()
    total_cost    = results_df["fuel_cost_eur"].sum()

    print(f"\n[RESULTS] ── Résumé journalier ──────────────────────")
    print(f"  Revenu brut     : {total_revenue:>12,.0f} €")
    print(f"  Coût production : {total_cost:>12,.0f} €")
    print(f"  Marge nette     : {total_margin:>12,.0f} €")
    print(f"  Production moy  : {results_df['total_prod_mw'].mean():>8,.1f} MW")

    for asset in ASSETS_DISPATCH:
        e_col = f"E_{asset}_mwh"
        if e_col in results_df.columns:
            total_e = results_df[e_col].sum()
            avg_p   = results_df[f"P_{asset}_mw"].mean()
            print(f"  {asset:<12} : {total_e:>8,.1f} MWh produits  "
                  f"| {avg_p:>6.1f} MW moy")

    return results_df

'''
# ─────────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────────
'''

def save_results(
        results_df: pd.DataFrame,
        output_dir: str = "data/processed",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"dispatch_results_{TARGET_DATE}.csv"
    filepath = os.path.join(output_dir, filename)
    results_df.to_csv(filepath, index=False)
    print(f"\n[SAVE] Résultats sauvegardés → {filepath}")
    return filepath

'''
# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
'''

def run_optimization(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Pipeline complet :
      load → build_model → solve → extract_results → save
    """
    print("=" * 60)
    print(f"OPTIMISATION DAY-AHEAD — {TARGET_DATE}")
    print("=" * 60)

    df               = load_optimizer_input(data_dir)
    prob, P, SOC, V_res, V_aval, P_pump = build_lp_model(df)
    status           = solve(prob)
    results_df = extract_results(df, P, SOC, V_res, V_aval, P_pump)
    save_results(results_df, data_dir)

    print("=" * 60)
    print("OPTIMISATION TERMINÉE")
    print("=" * 60)

    return results_df

'''
# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────
'''

if __name__ == "__main__":
    results = run_optimization()

    print("\nAperçu des 8 premiers slots :")
    cols = ["slot", "datetime", "da_price",
            "P_nuclear_mw", "P_gas_mw", "P_hydro_mw",
            "P_wind_mw", "P_solar_mw", "P_battery_mw",
            "soc_battery_mwh", "margin_eur"]
    print(results[cols].head(8).to_string(index=False))