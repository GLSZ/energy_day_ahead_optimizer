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

    hydro_budget = ASSETS["hydro_reservoir"]["daily_energy_budget"]

    prob += (
        pulp.lpSum(P["hydro"][t] * SLOT_DURATION_H for t in SLOTS) <= hydro_budget,
        "Hydro_daily_budget",
    )
    n_constraints += 1
    print(f"[LP] C3 Budget hydro : 1 contrainte (budget={hydro_budget} MWh/j)")

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
    # Pas de contrainte supplémentaire nécessaire ici.

    print(f"\n[LP] Total contraintes : {n_constraints}")
    print(f"[LP] Modèle prêt : {len(prob.variables())} variables, "
          f"{len(prob.constraints)} contraintes")

    return prob, P, SOC

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
    prob, P, SOC     = build_lp_model(df)
    status           = solve(prob)
    results_df       = extract_results(df, P, SOC)
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