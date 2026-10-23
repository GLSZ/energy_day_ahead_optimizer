# src/entsoe_client.py

"""
entsoe_client.py — Authentification ENTSO-E partagée.

Centralise ce qui était dupliqué à l'identique entre fetch_prices.py et
fetch_demand.py (Correction 5) : chargement du token depuis logs.env et
construction du client EntsoePandasClient. Les deux fichiers appelants
ont des besoins métier différents (prix vs charge), mais s'authentifient
de la même façon auprès de la même API — c'est cette partie technique
commune qui est centralisée ici, pas leur logique métier respective.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from entsoe import EntsoePandasClient

_ENV_PATH = Path(__file__).parent / "logs.env"
load_dotenv(_ENV_PATH)


def get_client() -> EntsoePandasClient:
    """
    Construit un client ENTSO-E authentifié à partir de la variable
    d'environnement ENTSOE_API_KEY (définie dans src/logs.env).

    Lève EnvironmentError si le token est absent.
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Token ENTSO-E manquant. "
            "Vérifie logs.env → ENTSOE_API_KEY=ton_token"
        )
    return EntsoePandasClient(api_key=api_key)
