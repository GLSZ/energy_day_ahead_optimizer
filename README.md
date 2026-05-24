# Energy Day-Ahead Optimizer

Optimisation du dispatch journalier d'un portefeuille de production
d'énergie face aux prix Day-Ahead du marché européen.

## Contexte

Ce projet simule les décisions d'un gestionnaire d'actifs de production
en marché Day-Ahead : pour chaque quart d'heure de la journée, quelle
puissance produire sur chaque actif (nucléaire, gaz CCGT, hydraulique,
éolien, solaire, batterie) afin de maximiser le profit ?

**Formulation** : Programmation Linéaire (LP) — solver CBC via PuLP  
**Données** : ENTSO-E Transparency Platform, Open-Meteo, yfinance (TTF)  
**Pas de temps** : 15 minutes (96 slots/jour)

## Stack technique

- `PuLP` + CBC — optimisation LP
- `entsoe-py` — prix Day-Ahead ENTSO-E
- `openmeteo-requests` — prévisions météo (vent, irradiation)
- `yfinance` — prix TTF gaz naturel
- `pandas` / `numpy` — traitement des données
- `matplotlib` — visualisation des résultats

## Structure du projet
energy-day-ahead-optimizer/
├── src/
│   ├── fetch_prices.py     # Extraction prix DA (ENTSO-E)
│   ├── fetch_weather.py    # Extraction météo (Open-Meteo)
│   ├── preprocess.py       # Alignement & calcul coût marginal gaz
│   ├── optimizer.py        # Modèle LP de dispatch (PuLP)
│   └── visualize.py        # Graphiques résultats
├── config.py               # Paramètres centralisés
├── main.py                 # Point d'entrée pipeline complet
├── requirements.txt
├── logs.env.example        # Template de configuration (sans clés)
└── README.md

## Installation

```bash
git clone https://github.com/ton-username/energy-day-ahead-optimizer.git
cd energy-day-ahead-optimizer
pip install -r requirements.txt
cp logs.env.example logs.env
# → Renseigne ta clé ENTSO-E dans logs.env
```

## Clé API ENTSO-E

Inscription gratuite sur https://transparency.entsoe.eu  
Token reçu par email sous 24h, à placer dans `logs.env` :

## Utilisation

```bash
# Pipeline complet
python main.py

# Ou étape par étape
python src/fetch_prices.py
python src/fetch_weather.py
python src/preprocess.py
python src/optimizer.py
python src/visualize.py
```

## Résultats

Le modèle produit 6 graphiques :
1. Prix Day-Ahead & Clean Spark Spread
2. Plan de dispatch empilé par actif
3. Mix de production journalière (MWh + facteur de charge)
4. Comportement de la batterie (puissance + SOC)
5. P&L : revenu, coût, marge cumulée
6. Merit Order & heatmap de dispatch

## Auteur

Ingénieur en Mathématiques Appliquées à la Finance  
Spécialiste marchés de l'énergie