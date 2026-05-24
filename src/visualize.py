# src/visualize.py

"""
visualize.py — Visualisation complète des résultats du dispatch Day-Ahead.

Graphiques générés (6 figures) :
  1. Prix Day-Ahead + Clean Spark Spread
  2. Plan de dispatch par actif (MW, stacked area)
  3. Production cumulée par actif (MWh, pie + bar)
  4. Batterie : puissance + état de charge (SOC)
  5. P&L : revenu, coût, marge par slot
  6. Merit order & analyse économique
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_DATE, ASSETS

# ─────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────
# Palette cohérente : chaque actif a sa couleur dans tous les graphiques

COLORS = {
    "nuclear"  : "#1E3A5F",   # bleu marine foncé
    "hydro"    : "#2196F3",   # bleu vif
    "wind"     : "#4CAF50",   # vert
    "solar"    : "#FFC107",   # jaune-ambre
    "battery"  : "#9C27B0",   # violet
    "gas"      : "#FF5722",   # orange-rouge (coûteux → chaud)
    "price"    : "#E91E63",   # rose vif
    "spread"   : "#00BCD4",   # cyan
    "revenue"  : "#43A047",   # vert foncé
    "cost"     : "#E53935",   # rouge
    "margin"   : "#1565C0",   # bleu roi
    "soc"      : "#7B1FA2",   # violet foncé
    "grid"     : "#ECEFF1",   # gris très clair
}

ASSET_LABELS = {
    "nuclear"  : "Nucléaire",
    "gas"      : "Gaz CCGT",
    "hydro"    : "Hydraulique",
    "wind"     : "Éolien",
    "solar"    : "Solaire",
    "battery"  : "Batterie",
}

# Style global matplotlib
plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.grid"         : True,
    "grid.color"        : COLORS["grid"],
    "grid.linewidth"    : 0.8,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.framealpha" : 0.9,
})

'''
# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────
'''

def _slots_to_hours(slots: pd.Series) -> np.ndarray:
    """
    Convertit les slots (0..95) en heures décimales (0.0..23.75).
    Ex : slot 8 → 2.0h (08:00), slot 9 → 2.25h (08:15)
    Utilisé pour l'axe X de tous les graphiques.
    """
    return slots.values * 0.25   # 1 slot = 0.25h


def _format_hour(x, pos):
    """Formateur d'axe X : 6.25 → '06:15'"""
    h = int(x)
    m = int((x - h) * 60)
    return f"{h:02d}:{m:02d}"


def _eur_formatter(x, pos):
    """Formateur axe Y en euros : 12500 → '12 500 €'"""
    return f"{x:,.0f} €".replace(",", " ")


def _mw_formatter(x, pos):
    return f"{x:,.0f} MW"


def load_results(data_dir: str = "data/processed") -> pd.DataFrame:
    """Charge le CSV de résultats produit par optimizer.py."""
    filepath = os.path.join(data_dir, f"dispatch_results_{TARGET_DATE}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Résultats introuvables : {filepath}\n"
            "Lance d'abord : python src/optimizer.py"
        )
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Paris")
    return df


def load_optimizer_input(data_dir: str = "data/processed") -> pd.DataFrame:
    """Charge le CSV optimizer_input pour les colonnes de spread."""
    filepath = os.path.join(data_dir, f"optimizer_input_{TARGET_DATE}.csv")
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Paris")
    return df


# ─────────────────────────────────────────────
# FIGURE 1 — Prix DA + Clean Spark Spread
# ─────────────────────────────────────────────

def plot_prices(df_res: pd.DataFrame, df_opt: pd.DataFrame, output_dir: str):
    """
    Double axe Y :
    - Axe gauche  : prix Day-Ahead (€/MWh) — courbe rose
    - Axe droit   : clean spark spread (€/MWh) — barres cyan
    - Ligne rouge : coût marginal gaz (seuil de rentabilité CCGT)

    Lecture : quand le spread est positif (barres au-dessus de 0),
    la centrale gaz est rentable. Quand il est négatif, elle doit s'arrêter.
    """
    hours = _slots_to_hours(df_res["slot"])
    gas_srmc = df_opt["gas_marginal_cost"].iloc[0]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"Prix Day-Ahead & Clean Spark Spread — {TARGET_DATE}",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Barres : clean spark spread (positif = vert, négatif = rouge)
    spread = df_opt["clean_spark_spread"].values
    bar_colors = [COLORS["revenue"] if s >= 0 else COLORS["cost"] for s in spread]
    ax1.bar(hours, spread, width=0.22, color=bar_colors, alpha=0.35,
            label="Clean Spark Spread (€/MWh)", zorder=2)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Clean Spark Spread (€/MWh)", color=COLORS["spread"])
    ax1.tick_params(axis="y", labelcolor=COLORS["spread"])

    # Courbe : prix DA sur axe droit
    ax2 = ax1.twinx()
    ax2.plot(hours, df_res["da_price"], color=COLORS["price"],
             linewidth=2.2, label="Prix DA (€/MWh)", zorder=3)
    ax2.axhline(gas_srmc, color=COLORS["gas"], linewidth=1.5,
                linestyle=":", label=f"SRMC gaz ({gas_srmc:.0f} €/MWh)")
    ax2.set_ylabel("Prix Day-Ahead (€/MWh)", color=COLORS["price"])
    ax2.tick_params(axis="y", labelcolor=COLORS["price"])

    # Axe X en heures
    ax1.set_xlabel("Heure de la journée")
    ax1.xaxis.set_major_formatter(FuncFormatter(_format_hour))
    ax1.set_xlim(-0.25, 24)
    ax1.set_xticks(range(0, 25, 2))

    # Légende combinée des deux axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Annotations : prix min / max
    idx_max = df_res["da_price"].idxmax()
    idx_min = df_res["da_price"].idxmin()
    ax2.annotate(
        f"Max\n{df_res['da_price'].max():.1f} €",
        xy=(hours[idx_max], df_res["da_price"].max()),
        xytext=(hours[idx_max] + 0.5, df_res["da_price"].max() + 2),
        fontsize=8, color=COLORS["price"],
        arrowprops=dict(arrowstyle="->", color=COLORS["price"], lw=1)
    )
    ax2.annotate(
        f"Min\n{df_res['da_price'].min():.1f} €",
        xy=(hours[idx_min], df_res["da_price"].min()),
        xytext=(hours[idx_min] + 0.5, df_res["da_price"].min() - 5),
        fontsize=8, color=COLORS["price"],
        arrowprops=dict(arrowstyle="->", color=COLORS["price"], lw=1)
    )

    plt.tight_layout()
    _save(fig, output_dir, "01_prix_spread")


# ─────────────────────────────────────────────
# FIGURE 2 — Plan de Dispatch (Stacked Area)
# ─────────────────────────────────────────────

def plot_dispatch(df_res: pd.DataFrame, output_dir: str):
    """
    Stacked area chart : chaque couche = production d'un actif (MW).

    L'ordre des couches suit le merit order (du moins cher au plus cher) :
    Solaire → Éolien → Nucléaire → Hydraulique → Batterie → Gaz

    La courbe rouge en surimpression = prix DA (axe droit) pour montrer
    la corrélation entre prix élevés et dispatch du gaz / décharge batterie.

    La zone de charge batterie (P < 0) est affichée séparément en bas.
    """
    hours  = _slots_to_hours(df_res["slot"])

    # Actifs producteurs (P ≥ 0) dans l'ordre des couches
    stack_assets = ["solar", "wind", "nuclear", "hydro", "battery", "gas"]

    # Sépare la batterie : décharge (positive) vs charge (négative)
    bat_discharge = df_res["P_battery_mw"].clip(lower=0)
    bat_charge    = df_res["P_battery_mw"].clip(upper=0)   # valeurs négatives

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True
    )
    fig.suptitle(
        f"Plan de Dispatch Day-Ahead — {TARGET_DATE}",
        fontsize=14, fontweight="bold"
    )

    # ── Partie haute : production (stacked area) ──────────────────────────
    bottoms = np.zeros(len(hours))
    for asset in stack_assets:
        col = f"P_{asset}_mw"
        if col not in df_res.columns:
            continue
        values = df_res[col].clip(lower=0).values   # garde uniquement P ≥ 0
        ax1.fill_between(hours, bottoms, bottoms + values,
                         color=COLORS[asset], alpha=0.85,
                         label=ASSET_LABELS[asset], step="pre")
        bottoms += values

    # Courbe prix DA sur axe droit
    ax_price = ax1.twinx()
    ax_price.plot(hours, df_res["da_price"], color=COLORS["price"],
                  linewidth=2, linestyle="--", label="Prix DA", alpha=0.8)
    ax_price.set_ylabel("Prix DA (€/MWh)", color=COLORS["price"])
    ax_price.tick_params(axis="y", labelcolor=COLORS["price"])

    ax1.set_ylabel("Puissance (MW)")
    ax1.set_title("Production par actif (empilée)", loc="left", fontsize=11)

    # Légende
    handles = [
        mpatches.Patch(color=COLORS[a], label=ASSET_LABELS[a])
        for a in stack_assets if f"P_{a}_mw" in df_res.columns
    ]
    ax1.legend(handles=handles, loc="upper left", ncol=3)

    # ── Partie basse : charge batterie (P < 0) ────────────────────────────
    # Montre quand la batterie consomme de l'énergie (charge depuis le réseau)
    ax2.fill_between(hours, bat_charge.values, 0,
                     color=COLORS["battery"], alpha=0.6,
                     label="Charge batterie", step="pre")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Charge (MW)", color=COLORS["battery"])
    ax2.set_title("Charge batterie", loc="left", fontsize=10)
    ax2.tick_params(axis="y", labelcolor=COLORS["battery"])

    ax2.set_xlabel("Heure de la journée")
    ax2.xaxis.set_major_formatter(FuncFormatter(_format_hour))
    ax2.set_xlim(-0.25, 24)
    ax2.set_xticks(range(0, 25, 2))

    plt.tight_layout()
    _save(fig, output_dir, "02_dispatch_stack")


# ─────────────────────────────────────────────
# FIGURE 3 — Production par actif (MWh)
# ─────────────────────────────────────────────

def plot_energy_mix(df_res: pd.DataFrame, output_dir: str):
    """
    Deux vues complémentaires de la production journalière :
    - Gauche : camembert (pie) → part de chaque actif dans le MIX
    - Droite : barres horizontales → MWh absolus + facteur de charge (%)

    Le facteur de charge (capacity factor) = production réelle / production max théorique
    = E_produite / (capacity_max × 24h)
    C'est une métrique clé pour comparer les technologies entre elles.
    """
    # Calcul des énergies journalières (MWh)
    energy = {}
    for asset in ASSET_LABELS:
        col = f"E_{asset}_mwh"
        if col in df_res.columns:
            total = df_res[col].sum()
            energy[asset] = max(total, 0)   # exclut la charge batterie (négatif)

    assets_sorted = sorted(energy, key=energy.get, reverse=True)
    values        = [energy[a] for a in assets_sorted]
    colors        = [COLORS[a] for a in assets_sorted]
    labels        = [ASSET_LABELS[a] for a in assets_sorted]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Mix de Production Journalière — {TARGET_DATE}",
        fontsize=14, fontweight="bold"
    )

    # ── Camembert ─────────────────────────────────────────────────────────
    wedges, texts, autotexts = ax_pie.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    total_mwh = sum(values)
    ax_pie.set_title(
        f"Mix énergétique\nTotal : {total_mwh:,.0f} MWh",
        fontsize=11
    )

    # ── Barres horizontales + facteur de charge ───────────────────────────
    y_pos = range(len(assets_sorted))
    bars  = ax_bar.barh(y_pos, values, color=colors, alpha=0.85, height=0.6)

    # Annotation : valeur MWh + facteur de charge
    for i, (asset, val) in enumerate(zip(assets_sorted, values)):
        cap_max = ASSETS.get(asset, {}).get("capacity_max", 1)
        cf      = val / (cap_max * 24) * 100   # facteur de charge en %
        ax_bar.text(
            val + 5, i,
            f"{val:,.0f} MWh  |  CF {cf:.1f}%",
            va="center", fontsize=9, color="#333333"
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel("Énergie produite (MWh)")
    ax_bar.set_title("Production & Facteur de charge", fontsize=11)
    ax_bar.set_xlim(0, max(values) * 1.4)
    ax_bar.invert_yaxis()   # actif le plus producteur en haut

    plt.tight_layout()
    _save(fig, output_dir, "03_energy_mix")


# ─────────────────────────────────────────────
# FIGURE 4 — Batterie : Puissance + SOC
# ─────────────────────────────────────────────

def plot_battery(df_res: pd.DataFrame, output_dir: str):
    """
    Double panel pour analyser le comportement de la batterie :
    - Haut : puissance (MW) — positif=décharge, négatif=charge
    - Bas  : état de charge SOC (MWh) avec les bornes min/max

    On superpose le prix DA pour vérifier la logique d'arbitrage :
    la batterie doit charger quand les prix sont BAS
    et décharger quand les prix sont HAUTS.
    C'est exactement ce que l'optimiseur doit trouver.
    """
    hours = _slots_to_hours(df_res["slot"])
    bat   = ASSETS["battery"]

    fig, (ax_pow, ax_soc) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 1]},
        sharex=True
    )
    fig.suptitle(
        f"Comportement de la Batterie — {TARGET_DATE}",
        fontsize=14, fontweight="bold"
    )

    # ── Puissance batterie ────────────────────────────────────────────────
    bat_power = df_res["P_battery_mw"].values
    discharge = np.clip(bat_power, 0, None)
    charge    = np.clip(bat_power, None, 0)

    ax_pow.fill_between(hours, 0, discharge,
                        color=COLORS["battery"], alpha=0.7,
                        label="Décharge (vente)", step="pre")
    ax_pow.fill_between(hours, charge, 0,
                        color=COLORS["cost"], alpha=0.5,
                        label="Charge (achat)", step="pre")
    ax_pow.axhline(0, color="black", linewidth=0.8)

    # Prix DA sur axe droit pour vérifier l'arbitrage
    ax_p = ax_pow.twinx()
    ax_p.plot(hours, df_res["da_price"], color=COLORS["price"],
              linewidth=1.8, linestyle="--", alpha=0.7, label="Prix DA")
    ax_p.set_ylabel("Prix DA (€/MWh)", color=COLORS["price"])
    ax_p.tick_params(axis="y", labelcolor=COLORS["price"])

    ax_pow.set_ylabel("Puissance (MW)", color=COLORS["battery"])
    ax_pow.set_ylim(bat["capacity_min"] * 1.2, bat["capacity_max"] * 1.2)
    ax_pow.set_title("Puissance batterie (décharge = vente, charge = achat)", loc="left")

    lines1, labels1 = ax_pow.get_legend_handles_labels()
    lines2, labels2 = ax_p.get_legend_handles_labels()
    ax_pow.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # ── État de charge (SOC) ──────────────────────────────────────────────
    soc_min_mwh  = bat["soc_min"] * bat["capacity_mwh"]   # 20 MWh
    soc_max_mwh  = bat["soc_max"] * bat["capacity_mwh"]   # 180 MWh
    soc_init_mwh = bat["soc_initial"] * bat["capacity_mwh"] # 100 MWh

    ax_soc.plot(hours, df_res["soc_battery_mwh"],
                color=COLORS["soc"], linewidth=2.2,
                label="SOC (MWh)", drawstyle="steps-pre")
    ax_soc.fill_between(hours, df_res["soc_battery_mwh"],
                        alpha=0.15, color=COLORS["soc"], step="pre")

    # Lignes de référence : bornes et état initial
    ax_soc.axhline(soc_max_mwh, color=COLORS["revenue"], linewidth=1.2,
                   linestyle="--", label=f"SOC max ({soc_max_mwh:.0f} MWh)")
    ax_soc.axhline(soc_min_mwh, color=COLORS["cost"], linewidth=1.2,
                   linestyle="--", label=f"SOC min ({soc_min_mwh:.0f} MWh)")
    ax_soc.axhline(soc_init_mwh, color="gray", linewidth=1,
                   linestyle=":", label=f"SOC initial ({soc_init_mwh:.0f} MWh)")

    # Zone colorée entre les bornes (zone de fonctionnement autorisée)
    ax_soc.fill_between(
        [0, 24], soc_min_mwh, soc_max_mwh,
        color=COLORS["grid"], alpha=0.5, label="Zone autorisée"
    )

    ax_soc.set_ylabel("Énergie stockée (MWh)", color=COLORS["soc"])
    ax_soc.set_ylim(0, bat["capacity_mwh"] * 1.05)
    ax_soc.set_title("État de charge (SOC)", loc="left")
    ax_soc.legend(loc="upper left", ncol=2)

    ax_soc.set_xlabel("Heure de la journée")
    ax_soc.xaxis.set_major_formatter(FuncFormatter(_format_hour))
    ax_soc.set_xlim(-0.25, 24)
    ax_soc.set_xticks(range(0, 25, 2))

    plt.tight_layout()
    _save(fig, output_dir, "04_battery")


# ─────────────────────────────────────────────
# FIGURE 5 — P&L : Revenu / Coût / Marge
# ─────────────────────────────────────────────

def plot_pnl(df_res: pd.DataFrame, output_dir: str):
    """
    Analyse financière slot par slot :
    - Barres empilées : revenu brut (vert) et coût de production (rouge)
    - Courbe : marge nette cumulée (€) — permet de voir à quelle heure
      le profit journalier est généré (souvent pic matin et soir)

    Résumé en bas : KPIs financiers de la journée.
    """
    hours = _slots_to_hours(df_res["slot"])

    total_rev    = df_res["revenue_eur"].sum()
    total_cost   = df_res["fuel_cost_eur"].sum()
    total_margin = df_res["margin_eur"].sum()
    margin_cum   = df_res["margin_eur"].cumsum()

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_bar  = fig.add_subplot(gs[0, :])    # graphique principal pleine largeur
    ax_cum  = fig.add_subplot(gs[1, :2])   # marge cumulée
    ax_kpi  = fig.add_subplot(gs[1, 2])    # KPI cards

    fig.suptitle(
        f"P&L — Revenu, Coût & Marge — {TARGET_DATE}",
        fontsize=14, fontweight="bold"
    )

    # ── Barres : revenu et coût par slot ─────────────────────────────────
    width = 0.20
    ax_bar.bar(hours - width/2, df_res["revenue_eur"], width=width,
               color=COLORS["revenue"], alpha=0.8, label="Revenu brut")
    ax_bar.bar(hours + width/2, df_res["fuel_cost_eur"], width=width,
               color=COLORS["cost"], alpha=0.8, label="Coût production")

    # Marge par slot : ligne
    ax_bar_m = ax_bar.twinx()
    ax_bar_m.plot(hours, df_res["margin_eur"], color=COLORS["margin"],
                  linewidth=2, label="Marge nette (€)")
    ax_bar_m.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_bar_m.set_ylabel("Marge par slot (€)", color=COLORS["margin"])
    ax_bar_m.tick_params(axis="y", labelcolor=COLORS["margin"])

    ax_bar.set_ylabel("€ par slot")
    ax_bar.set_title("Revenu & Coût par slot de 15min", loc="left")
    ax_bar.xaxis.set_major_formatter(FuncFormatter(_format_hour))
    ax_bar.set_xlim(-0.25, 24)

    lines1, labels1 = ax_bar.get_legend_handles_labels()
    lines2, labels2 = ax_bar_m.get_legend_handles_labels()
    ax_bar.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # ── Marge cumulée ────────────────────────────────────────────────────
    # Montre à quelle heure le profit est généré dans la journée
    ax_cum.fill_between(hours, 0, margin_cum,
                        where=margin_cum >= 0,
                        color=COLORS["margin"], alpha=0.25)
    ax_cum.fill_between(hours, 0, margin_cum,
                        where=margin_cum < 0,
                        color=COLORS["cost"], alpha=0.25)
    ax_cum.plot(hours, margin_cum, color=COLORS["margin"], linewidth=2.2)
    ax_cum.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax_cum.yaxis.set_major_formatter(FuncFormatter(_eur_formatter))
    ax_cum.set_xlabel("Heure de la journée")
    ax_cum.set_ylabel("Marge cumulée (€)")
    ax_cum.set_title("Marge cumulée au fil de la journée", loc="left")
    ax_cum.xaxis.set_major_formatter(FuncFormatter(_format_hour))
    ax_cum.set_xlim(-0.25, 24)

    # Annotation : marge finale
    ax_cum.annotate(
        f"Total\n{total_margin:,.0f} €",
        xy=(23.75, margin_cum.iloc[-1]),
        xytext=(20, margin_cum.iloc[-1] * 0.7),
        fontsize=9, fontweight="bold", color=COLORS["margin"],
        arrowprops=dict(arrowstyle="->", color=COLORS["margin"])
    )

    # ── KPI Cards ────────────────────────────────────────────────────────
    ax_kpi.axis("off")   # pas d'axes, juste du texte

    kpis = [
        ("Revenu brut",     f"{total_rev:,.0f} €",    COLORS["revenue"]),
        ("Coût production", f"{total_cost:,.0f} €",   COLORS["cost"]),
        ("Marge nette",     f"{total_margin:,.0f} €", COLORS["margin"]),
        ("Marge / MWh",
         f"{total_margin / max(df_res['E_nuclear_mwh'].sum() + df_res['E_gas_mwh'].sum(), 1):,.1f} €/MWh",
         COLORS["margin"]),
    ]

    for i, (label, value, color) in enumerate(kpis):
        y = 0.85 - i * 0.22
        ax_kpi.text(0.05, y + 0.06, label, transform=ax_kpi.transAxes,
                    fontsize=9, color="gray")
        ax_kpi.text(0.05, y, value, transform=ax_kpi.transAxes,
                    fontsize=15, fontweight="bold", color=color)
        ax_kpi.axhline(y - 0.04, color=COLORS["grid"], linewidth=1)#,
                       #transform=ax_kpi.get_xaxis_transform())

    ax_kpi.set_title("KPIs journaliers", loc="left", fontsize=11)

    _save(fig, output_dir, "05_pnl")


# ─────────────────────────────────────────────
# FIGURE 6 — Merit Order & Analyse économique
# ─────────────────────────────────────────────

def plot_merit_order(df_res: pd.DataFrame, df_opt: pd.DataFrame, output_dir: str):
    """
    Deux analyses complémentaires :
    - Gauche : courbe de merit order (MW cumulés vs coût marginal)
      Visualise l'ordre dans lequel les centrales entrent en production
    - Droite : heatmap de la production par actif et par heure
      Vue synthétique du plan de dispatch sur 24h

    La merit order est fondamentale en économie de l'énergie :
    le prix de marché = coût marginal de la dernière centrale appelée
    (la "price setter"). C'est le mécanisme qui fixe le prix spot.
    """
    fig, (ax_mo, ax_hm) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Merit Order & Heatmap Dispatch — {TARGET_DATE}",
        fontsize=14, fontweight="bold"
    )

    # ── Merit Order ───────────────────────────────────────────────────────
    # On classe les actifs par coût marginal croissant
    gas_srmc = df_opt["gas_marginal_cost"].iloc[0]

    merit = [
        ("Éolien",      0.0,     ASSETS["wind"]["capacity_max"],       COLORS["wind"]),
        ("Solaire",     0.0,     ASSETS["solar"]["capacity_max"],      COLORS["solar"]),
        ("Hydraulique", 5.0,     ASSETS["hydro_reservoir"]["capacity_max"], COLORS["hydro"]),
        ("Batterie",    2.0,     ASSETS["battery"]["capacity_max"],    COLORS["battery"]),
        ("Nucléaire",   12.0,    ASSETS["nuclear"]["capacity_max"],    COLORS["nuclear"]),
        ("Gaz CCGT",    gas_srmc, ASSETS["gas_ccgt"]["capacity_max"],  COLORS["gas"]),
    ]
    # Tri par coût marginal
    merit = sorted(merit, key=lambda x: x[1])

    cumulative_mw = 0
    for label, cost, capacity, color in merit:
        ax_mo.barh(
            cost, capacity,
            left=cumulative_mw,
            height=max(cost * 0.08, 3),   # hauteur proportionnelle au coût
            color=color, alpha=0.85,
            label=f"{label} ({cost:.0f} €/MWh)"
        )
        ax_mo.text(
            cumulative_mw + capacity / 2, cost + 1.5,
            label, ha="center", va="bottom", fontsize=8, fontweight="bold"
        )
        cumulative_mw += capacity

    # Prix moyen DA comme ligne de référence
    avg_price = df_res["da_price"].mean()
    ax_mo.axhline(avg_price, color=COLORS["price"], linewidth=2,
                  linestyle="--", label=f"Prix moy DA ({avg_price:.1f} €/MWh)")

    ax_mo.set_xlabel("Capacité cumulée (MW)")
    ax_mo.set_ylabel("Coût marginal (€/MWh)")
    ax_mo.set_title("Courbe de Merit Order", loc="left")
    ax_mo.legend(loc="upper left", fontsize=8)
    ax_mo.set_xlim(0, cumulative_mw * 1.05)
    ax_mo.set_ylim(0, gas_srmc * 1.2)

    # ── Heatmap dispatch ──────────────────────────────────────────────────
    # Matrice : actifs (lignes) × heures agrégées (colonnes)
    # On agrège par heure complète (groupby sur floor(slot/4))
    assets_hm = ["solar", "wind", "nuclear", "hydro", "battery", "gas"]

    # Agrégation par heure (moyenne des 4 slots de 15min)
    df_res["hour"] = df_res["slot"] // 4
    hourly = df_res.groupby("hour")[
        [f"P_{a}_mw" for a in assets_hm if f"P_{a}_mw" in df_res.columns]
    ].mean()

    matrix = hourly.T.values   # shape (n_assets, 24)
    matrix = np.clip(matrix, 0, None)   # pas de valeurs négatives pour la heatmap

    im = ax_hm.imshow(
        matrix,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax_hm.set_yticks(range(len(assets_hm)))
    ax_hm.set_yticklabels([ASSET_LABELS[a] for a in assets_hm])
    ax_hm.set_xticks(range(0, 24, 2))
    ax_hm.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)], fontsize=8)
    ax_hm.set_xlabel("Heure de la journée")
    ax_hm.set_title("Heatmap — Puissance par actif (MW moyen/h)", loc="left")

    plt.colorbar(im, ax=ax_hm, label="MW", shrink=0.8)

    plt.tight_layout()
    _save(fig, output_dir, "06_merit_order_heatmap")

'''
# ─────────────────────────────────────────────
# UTILITAIRE : SAUVEGARDE
# ─────────────────────────────────────────────
'''

def _save(fig: plt.Figure, output_dir: str, name: str):
    """Sauvegarde en PNG haute résolution et affiche."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}_{TARGET_DATE}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"[SAVE] {filepath}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_visualization(
    results_dir: str = "data/processed",
    output_dir:  str = "outputs/figures",
):
    print("=" * 60)
    print(f"VISUALISATION — {TARGET_DATE}")
    print("=" * 60)

    df_res = load_results(results_dir)
    df_opt = load_optimizer_input(results_dir)

    plot_prices(df_res, df_opt, output_dir)
    plot_dispatch(df_res, output_dir)
    plot_energy_mix(df_res, output_dir)
    plot_battery(df_res, output_dir)
    plot_pnl(df_res, output_dir)
    plot_merit_order(df_res, df_opt, output_dir)

    print("=" * 60)
    print(f"VISUALISATION TERMINÉE — 6 figures dans {output_dir}/")
    print("=" * 60)


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_visualization()