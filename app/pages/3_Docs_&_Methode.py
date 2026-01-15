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
- **Mode cible**: on pénalise l'écart aux cibles UCS/Slump.
- Le score sert uniquement à **ordonner** les candidats, pas à certifier une recette.
"""
    )

    st.markdown(
        """
### Comment utiliser le générateur de recettes
1) Choisir le **dataset** (WW / L01 OLD / L01 NEW).
2) Définir l’objectif :
   - **Contraintes min** (Slump ≥ X, UCS ≥ Y), ou
   - **Cible + tolérance** (ex: UCS 1200 ± 100).
3) Indiquer **N candidats** (plus N est grand, plus on explore).
4) Choisir **Search mode** :
   - **Uniform** : tirage entre min/max.
   - **Bootstrap** : tirage dans les valeurs observées (plus réaliste).
5) Cliquer **Générer** → lire le tableau **Top K**.
6) Télécharger l’Excel si besoin.

Astuce : si aucun candidat ne passe, augmenter N ou relâcher les seuils.
"""
    )

    st.markdown(
        """
### Focus UCS (métriques clés)
Le projet vise d’abord la **résistance UCS**, donc on surveille surtout :
- **R² UCS** : plus c’est proche de 1, mieux le modèle explique la variance.
- **RMSE UCS (kPa)** : erreur moyenne typique (plus petit = mieux).

Valeurs indicatives (modèles finaux):
- **FINAL_old_best** : UCS L01 R² ≈ 0.46, RMSE ≈ 359 kPa; UCS WW R² ≈ 0.94, RMSE ≈ 169 kPa.
- **FINAL_new_best** : UCS L01 R² ≈ 0.86, RMSE ≈ 230 kPa; UCS WW R² ≈ 0.93, RMSE ≈ 174 kPa.

"""
    )


if __name__ == "__main__":
    main()
