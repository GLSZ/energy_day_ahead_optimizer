# main.py

"""
main.py — Point d'entrée principal du projet
Day-Ahead Production Asset Optimizer.

Pipeline complet :
  1. Extraction des prix DA (ENTSO-E)
  2. Extraction météo + conversion puissance (Open-Meteo)
  3. Preprocessing (alignement, coût marginal gaz, DataFrame optimiseur)
  4. Optimisation LP (dispatch Day-Ahead — PuLP/CBC)
  5. Visualisation (6 figures)

Usage :
  python main.py                  # pipeline complet
  python main.py --skip-fetch     # repart des CSV existants
  python main.py --only-viz       # régénère uniquement les figures
  python main.py --date 2024-03-15 # optimise une date spécifique
"""

import sys
import os
import argparse
import traceback
from datetime import datetime, date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import TARGET_DATE, ASSETS

#from config import TARGET_DATE, ASSETS, PATHS if hasattr(__import__('config'), 'PATHS') else None


# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────

def _header(title: str):
    width = 65
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

def _step(n: int, total: int, label: str):
    print(f"\n{'─' * 65}")
    print(f"  ÉTAPE {n}/{total} — {label}")
    print(f"{'─' * 65}")

def _success(msg: str): print(f"  ✓ {msg}")
def _warn(msg: str):    print(f"  ⚠ {msg}")
def _error(msg: str):   print(f"  ✗ {msg}")

def _ensure_dirs():
    for d in ["data/raw", "data/processed", "outputs/figures"]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# ARGUMENTS CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Day-Ahead Production Asset Optimizer"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Saute fetch_prices + fetch_weather (repart des CSV)"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Saute le preprocessing (repart de optimizer_input existant)"
    )
    parser.add_argument(
        "--only-viz",
        action="store_true",
        help="Régénère uniquement les figures (dispatch_results doit exister)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Saute la visualisation"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date à optimiser au format YYYY-MM-DD (défaut : TARGET_DATE dans config.py)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# ÉTAPES DU PIPELINE
# ─────────────────────────────────────────────

def step_fetch_prices(target_date) -> None:
    from fetch_prices import fetch_day_ahead_prices, save_prices
    prices = fetch_day_ahead_prices(target_date=target_date)
    save_prices(prices)
    n = len(prices)
    _success(f"{n} slots récupérés — "
             f"min {prices.min():.2f} €/MWh  "
             f"max {prices.max():.2f} €/MWh  "
             f"moy {prices.mean():.2f} €/MWh")


def step_fetch_weather(target_date) -> None:
    from fetch_weather import fetch_weather_power, save_weather
    df = fetch_weather_power(target_date=target_date)
    save_weather(df)
    _success(f"Météo récupérée — {len(df)} slots")
    _success(f"Éolien  : {df['wind_power_mw'].mean():.1f} MW moy  "
             f"| {df['wind_power_mw'].sum() * 0.25:.0f} MWh")
    _success(f"Solaire : {df['solar_power_mw'].mean():.1f} MW moy  "
             f"| {df['solar_power_mw'].sum() * 0.25:.0f} MWh")


def step_preprocess() -> object:
    from preprocess import run_preprocessing
    df = run_preprocessing()
    _success(f"DataFrame optimiseur : {len(df)} slots × {len(df.columns)} colonnes")
    _success(f"Clean spark spread moy : {df['clean_spark_spread'].mean():.2f} €/MWh")
    _success(f"Slots gaz rentable     : {df['gas_profitable'].sum()}/96")
    return df


def step_optimize() -> object:
    from optimizer import run_optimization
    results = run_optimization()
    _success(f"Statut solver    : Optimal ✓")
    _success(f"Marge journalière : {results['margin_eur'].sum():,.0f} €")
    _success(f"Revenu brut       : {results['revenue_eur'].sum():,.0f} €")
    _success(f"Coût production   : {results['fuel_cost_eur'].sum():,.0f} €")

    # Production par actif
    assets = ["nuclear", "gas", "hydro", "wind", "solar", "battery"]
    for a in assets:
        col = f"E_{a}_mwh"
        if col in results.columns:
            total = results[col].sum()
            _success(f"  {a:<12} : {total:>8,.1f} MWh")
    return results


def step_visualize() -> None:
    from visualize import run_visualization
    run_visualization()
    _success("6 figures générées dans outputs/figures/")


# ─────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────

def print_final_report(
    t_start: datetime,
    target_date,
    steps_ok: list,
    steps_ko: list,
):
    t_end    = datetime.now()
    duration = (t_end - t_start).total_seconds()
    minutes  = int(duration // 60)
    seconds  = int(duration % 60)

    _header("RAPPORT D'EXÉCUTION")
    print(f"\n  Projet    : Day-Ahead Production Asset Optimizer")
    print(f"  Date      : {target_date}")
    print(f"  Durée     : {minutes}min {seconds}s")
    print()

    print(f"  Étapes réussies ({len(steps_ok)}) :")
    for s in steps_ok:
        print(f"    ✓ {s}")

    if steps_ko:
        print(f"\n  Étapes échouées ({len(steps_ko)}) :")
        for s, err in steps_ko:
            print(f"    ✗ {s} : {err}")

    # Liste les figures générées
    fig_dir = "outputs/figures"
    if os.path.isdir(fig_dir):
        figs = sorted([f for f in os.listdir(fig_dir) if f.endswith(".png")])
        if figs:
            print(f"\n  Figures ({len(figs)}) :")
            for f in figs:
                size_kb = os.path.getsize(os.path.join(fig_dir, f)) // 1024
                print(f"    {f}  ({size_kb} Ko)")

    print(f"\n{'═' * 65}\n")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    args    = parse_args()
    t_start = datetime.now()

    # Résolution de la date cible
    # --date en CLI prend le dessus sur TARGET_DATE dans config.py
    if args.date:
        from datetime import date as dt_date
        target_date = dt_date.fromisoformat(args.date)
        # Patch dynamique de config pour que tous les modules utilisent cette date
        import config
        config.TARGET_DATE = target_date
    else:
        target_date = TARGET_DATE

    _header(
        f"DAY-AHEAD PRODUCTION ASSET OPTIMIZER\n"
        f"  Date cible : {target_date}  |  "
        f"Zone : France (RTE)"
    )

    _ensure_dirs()

    steps_ok = []
    steps_ko = []
    TOTAL    = 5

    # ── Mode --only-viz ───────────────────────────────────────────────────
    if args.only_viz:
        _warn("Mode --only-viz : régénération des figures uniquement")
        try:
            _step(5, 5, "Visualisation")
            step_visualize()
            steps_ok.append("Visualisation")
        except Exception as e:
            _error(str(e))
            traceback.print_exc()
            steps_ko.append(("Visualisation", str(e)))
        print_final_report(t_start, target_date, steps_ok, steps_ko)
        return

    # ─────────────────────────────────────────────────────────────────────
    # ÉTAPE 1 — FETCH PRIX DA (ENTSO-E)
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_fetch:
        _step(1, TOTAL, "Extraction des prix Day-Ahead (ENTSO-E)")
        try:
            step_fetch_prices(target_date)
            steps_ok.append("Fetch prix DA")
        except Exception as e:
            _error(str(e))
            traceback.print_exc()
            steps_ko.append(("Fetch prix DA", str(e)))
            print("\n  Pipeline interrompu — les prix DA sont indispensables.")
            print_final_report(t_start, target_date, steps_ok, steps_ko)
            sys.exit(1)
    else:
        _warn("Étape 1 (fetch prix) ignorée — --skip-fetch")
        steps_ok.append("Fetch prix DA (ignoré)")

    # ─────────────────────────────────────────────────────────────────────
    # ÉTAPE 2 — FETCH MÉTÉO (OPEN-METEO)
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_fetch:
        _step(2, TOTAL, "Extraction météo + conversion puissance (Open-Meteo)")
        try:
            step_fetch_weather(target_date)
            steps_ok.append("Fetch météo")
        except Exception as e:
            _error(str(e))
            traceback.print_exc()
            steps_ko.append(("Fetch météo", str(e)))
            # Non bloquant : on peut continuer sans météo
            # (les renouvelables seront à 0 MW)
            _warn("Continuation sans données météo — renouvelables à 0 MW")
    else:
        _warn("Étape 2 (fetch météo) ignorée — --skip-fetch")
        steps_ok.append("Fetch météo (ignoré)")

    # ─────────────────────────────────────────────────────────────────────
    # ÉTAPE 3 — PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_preprocess:
        _step(3, TOTAL, "Preprocessing — alignement & coût marginal gaz")
        try:
            step_preprocess()
            steps_ok.append("Preprocessing")
        except Exception as e:
            _error(str(e))
            traceback.print_exc()
            steps_ko.append(("Preprocessing", str(e)))
            print("\n  Pipeline interrompu — optimizer_input indispensable.")
            print_final_report(t_start, target_date, steps_ok, steps_ko)
            sys.exit(1)
    else:
        _warn("Étape 3 (preprocessing) ignorée — --skip-preprocess")
        steps_ok.append("Preprocessing (ignoré)")

    # ─────────────────────────────────────────────────────────────────────
    # ÉTAPE 4 — OPTIMISATION LP
    # ─────────────────────────────────────────────────────────────────────
    _step(4, TOTAL, "Optimisation LP — Dispatch Day-Ahead (PuLP/CBC)")
    try:
        step_optimize()
        steps_ok.append("Optimisation LP")
    except Exception as e:
        _error(str(e))
        traceback.print_exc()
        steps_ko.append(("Optimisation LP", str(e)))
        print("\n  Pipeline interrompu — les résultats sont indispensables à la viz.")
        print_final_report(t_start, target_date, steps_ok, steps_ko)
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────
    # ÉTAPE 5 — VISUALISATION
    # ─────────────────────────────────────────────────────────────────────
    if not args.no_viz:
        _step(5, TOTAL, "Visualisation — Dashboard complet (6 figures)")
        try:
            step_visualize()
            steps_ok.append("Visualisation (6 figures)")
        except Exception as e:
            _error(str(e))
            traceback.print_exc()
            steps_ko.append(("Visualisation", str(e)))
    else:
        _warn("Étape 5 (visualisation) ignorée — --no-viz")
        steps_ok.append("Visualisation (ignoré)")

    # ─────────────────────────────────────────────────────────────────────
    # RAPPORT FINAL
    # ─────────────────────────────────────────────────────────────────────
    print_final_report(t_start, target_date, steps_ok, steps_ko)
    sys.exit(0 if not steps_ko else 1)


if __name__ == "__main__":
    main()