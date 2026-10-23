"""
Tests de non-regression pour la Correction 1 (bug Q_spill / bassin aval)
et pour le bug de mapping ASSETS_DISPATCH -> config.ASSETS qui bloquait
la meme date.

Donnees utilisees : data/processed/optimizer_input_2026-09-13.csv, un
fichier REEL deja present dans le repo (pas de donnees synthetiques).
C'est exactement la date qui etait "Infeasible" avant les deux corrections.

Historique du bug :
  1) Q_spill (deversement du bassin aval) etait cree comme variable PuLP
     mais jamais soustraite dans le bilan V_aval -> aucun effet sur le
     modele (src/optimizer.py, bilan "Hydro_aval_balance_t").
  2) ASSETS_DISPATCH utilise les noms courts "gas"/"hydro" alors que
     config.ASSETS utilise les cles "gas_ccgt"/"hydro_reservoir" ->
     P_gas et P_hydro etaient creees avec capacity_min=capacity_max=0,
     ce qui contredisait la contrainte Hydro_env_flow (debit reserve).
Les deux bugs combines rendaient le solveur CBC "Infeasible" sur cette
date. Ce fichier verifie que les deux sont corriges et le restent.
"""
import datetime

import pulp
import pytest

import config
import optimizer

KNOWN_DATE = datetime.date(2026, 9, 13)  # anciennement Infeasible


@pytest.fixture
def optimizer_input_df():
    """Charge le vrai CSV optimizer_input pour la date de reference."""
    config.TARGET_DATE = KNOWN_DATE
    return optimizer.load_optimizer_input()


@pytest.fixture
def built_model(optimizer_input_df):
    """Construit le modele LP (sans le resoudre) pour les tests unitaires."""
    return optimizer.build_lp_model(optimizer_input_df)


def test_Q_spill_is_referenced_in_aval_balance(built_model):
    """
    Regression directe sur le bug : avant le fix, Q_spill[t] n'apparaissait
    dans AUCUNE contrainte du modele (variable creee mais jamais utilisee).
    Ce test echoue si le terme '- Q_spill[t]' est de nouveau retire du
    bilan V_aval dans build_lp_model().
    """
    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = built_model

    # ATTENTION PuLP : LpVariable surcharge `__eq__` (utilise pour ecrire
    # des contraintes du type `x == 5`), qui renvoie un objet LpConstraint
    # et non un booleen. Un test `variable in constraint` ou `variable ==
    # autre_variable` est donc TOUJOURS vrai au sens Python, meme si la
    # variable n'est pas reellement dans la contrainte (faux positif
    # silencieux). Il faut comparer par IDENTITE (`is`) sur les cles
    # reelles du dict (`constraint.keys()`), jamais par `in`/`==` direct.
    checked = 0
    for t in range(0, optimizer.N_SLOTS - 1, 10):
        name = f"Hydro_aval_balance_{t}"
        assert name in prob.constraints, f"contrainte '{name}' absente du modele"
        constraint = prob.constraints[name]
        is_present = any(v is Q_spill[t] for v in constraint.keys())
        assert is_present, (
            f"Q_spill[{t}] n'apparait pas dans '{name}' : la variable de "
            "deversement n'a aucun effet sur le bilan du bassin aval."
        )
        checked += 1
    assert checked > 0


def test_gas_and_hydro_have_nonzero_capacity(built_model):
    """
    Regression sur le second bug (mapping ASSET_CONFIG_KEY) : avant le fix,
    P_gas et P_hydro etaient creees avec upBound=0 (capacity_max=0) car
    "gas"/"hydro" ne correspondaient a aucune cle de config.ASSETS.
    """
    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = built_model

    assert P["gas"][0].upBound > 0, "P_gas plafonnee a 0 MW (bug de mapping de cle)"
    assert P["hydro"][0].upBound > 0, "P_hydro plafonnee a 0 MW (bug de mapping de cle)"


def test_solver_status_is_optimal_on_known_infeasible_date(built_model):
    """
    Avant les deux corrections : statut CBC = "Infeasible" sur cette date
    (confirme par execution reelle de main.py --date 2026-09-13).
    Apres correction : le modele doit se resoudre en "Optimal".
    """
    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = built_model
    status = pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))]
    assert status == "Optimal", (
        f"statut inattendu : {status} (regression sur le fix Q_spill / "
        "ASSET_CONFIG_KEY ?)"
    )


def test_hydro_reservoir_levels_stay_within_bounds(built_model):
    """
    Verifie qu'apres resolution, V_res et V_aval restent dans les bornes
    physiques definies dans config.HYDRO (le solveur ne peut de toute
    facon pas violer les bornes d'une LpVariable, mais on verifie ici
    qu'une solution existe bien et que les valeurs extraites sont
    coherentes avec les bornes attendues).
    """
    prob, P, SOC, V_res, V_aval, P_pump, Q_spill, U_gas, Start_gas = built_model
    status = pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))]
    assert status == "Optimal"

    res_min, res_max = config.HYDRO["reservoir"]["level_env_min_hm3"], config.HYDRO["reservoir"]["level_max_hm3"]
    aval_min, aval_max = config.HYDRO["lower_basin"]["level_env_min_hm3"], config.HYDRO["lower_basin"]["level_max_hm3"]

    for t in range(optimizer.N_SLOTS):
        v_res = pulp.value(V_res[t])
        v_aval = pulp.value(V_aval[t])
        assert res_min - 1e-6 <= v_res <= res_max + 1e-6, f"V_res[{t}]={v_res} hors bornes"
        assert aval_min - 1e-6 <= v_aval <= aval_max + 1e-6, f"V_aval[{t}]={v_aval} hors bornes"
