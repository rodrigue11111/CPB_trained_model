# -*- coding: utf-8 -*-
"""Page Documentation et méthode Streamlit."""

from __future__ import annotations

import streamlit as st

from app.ui.components import load_css


def main() -> None:
    st.title("Docs & Méthode")
    load_css("app/ui/styles.css")

    st.markdown(
        """
### Pourquoi 3 datasets ?
- **WW** et **L01 OLD** viennent de campagnes différentes et ont des distributions distinctes.
- **L01 NEW** a un schéma enrichi (muscovite_added/total + muscovite_ratio).
- Le modèle sélectionne dynamiquement les features disponibles, sans forcer un schéma unique.

### Sécurité des données (important)
- Les fichiers Excel bruts (**data/**) ne sont **pas** embarqués sur Streamlit Cloud.
- À la place, l’application utilise un **profil léger** (`data_profiles.json`) :
  - **statistiques** (min/max/moyenne/écart‑type)
  - **catégories** (ex: Binder)
  - **échantillon partiel** (~50% des valeurs) pour le bootstrap léger
- Cela protège les données sensibles tout en gardant un comportement réaliste.

**Impact sur le programme**
- Le **Predictor** reste fiable (modèles chargés depuis joblib).
- Le **Générateur** :
  - en local (avec data/): bootstrap "complet"
  - sur le cloud (sans data/): bootstrap "léger" basé sur l’échantillon
- Les résultats restent cohérents, mais le bootstrap cloud est **moins riche** que le bootstrap local.

### Fit mode `hybrid_by_tailings`
- Un modèle **WW** et un modèle **L01** distincts.
- Permet d’optimiser chaque famille de tailings sans mélange des distributions.

### Bootstrap vs Uniform (sampling)
- **Uniform**: tirage aléatoire entre min et max observés.
- **Bootstrap**: tirage parmi les valeurs observées (plus réaliste).

### Tuning vs Fixed Params
- **Tuning**: recherche d’hyperparamètres (RandomizedSearchCV).
- **Fixed params**: réutilisation des meilleurs paramètres d’un sweep pour reproduire un run.

### R² et RMSE
- **R²** proche de 1: bonne explication de la variance.
- **RMSE**: erreur moyenne en unité cible (kPa / mm).
- Dans l’app, le ± affiché correspond au RMSE (approx.), **pas** un intervalle statistique.

IMPORTANT: **le modèle guide, le labo confirme**.
"""
    )

    st.markdown(
        """
### Pipeline (schéma simplifié)
```
Excel -> Nettoyage -> Features -> Entraînement -> Métriques
                             -> Optimisation -> Top_Recipes.xlsx
```
"""
    )

    st.markdown(
        """
### Score de classement (générateur)
- **Mode contraintes min**: score = UCS_pred - 0.02 * E/C - 0.02 * Ad % (si présents).
- **Mode cible**: on pénalise l’écart aux cibles UCS/Slump.
- Le score sert uniquement à **ordonner** les candidats, pas à certifier une recette.
"""
    )


if __name__ == "__main__":
    main()
