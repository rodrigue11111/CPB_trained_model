# -*- coding: utf-8 -*-
'''Page Documentation et méthode Streamlit.'''

from __future__ import annotations

import streamlit as st

from app.ui.components import load_css


def main() -> None:
    st.title("Docs & Méthode")
    load_css("app/ui/styles.css")

    st.markdown(
        '''
### Pourquoi 3 datasets ?
- **WW** et **L01 OLD** viennent de campagnes différentes et ont des distributions distinctes.
- **L01 NEW** a un schéma enrichi (muscovite_added/total + muscovite_ratio).
- Le modèle sélectionne dynamiquement les features disponibles, sans forcer un schéma unique.

### Fit mode `hybrid_by_tailings`
- Un modèle **WW** et un modèle **L01** distincts.
- Permet d'optimiser chaque famille de tailings sans mélange des distributions.

### Bootstrap vs Uniform (sampling)
- **Uniform**: tirage aléatoire entre min et max observés.
- **Bootstrap**: tirage parmi les valeurs observées (plus réaliste).

### Tuning vs Fixed Params
- **Tuning**: recherche d'hyperparamètres (RandomizedSearchCV).
- **Fixed params**: réutilisation des meilleurs paramètres d'un sweep pour reproduire un run.

### R2 et RMSE
- **R2** proche de 1: bonne explication de la variance.
- **RMSE**: erreur moyenne en unité cible (kPa / mm).
- Dans l'app, le +/- affiché correspond au RMSE (approx.), **pas** un intervalle statistique.

IMPORTANT: **le modèle guide, le labo confirme**.
'''
    )

    st.markdown(
        '''
### Pipeline (schéma simplifié)
```
Excel -> Nettoyage -> Features -> Entraînement -> Métriques
                             -> Optimisation -> Top_Recipes.xlsx
```
'''
    )

    st.markdown(
        '''
### Score de classement (générateur)
- **Mode contraintes min**: score = UCS_pred - 0.02 * E/C - 0.02 * Ad % (si présents).
- **Mode cible**: on pénalise l'écart aux cibles UCS/Slump.
- Le score sert uniquement à **ordonner** les candidats, pas à certifier une recette.
'''
    )

    st.markdown(
        '''
### Comment utiliser le générateur de recettes
1) Choisir le **dataset** (WW / L01 OLD / L01 NEW).
2) Définir l'objectif :
   - **Contraintes min** (Slump >= X, UCS >= Y), ou
   - **Cible + tolérance** (ex: UCS 1200 +/- 100).
3) Indiquer **N candidats** (plus N est grand, plus on explore).
4) Choisir **Search mode** :
   - **Uniform** : tirage entre min/max.
   - **Bootstrap** : tirage dans les valeurs observées (plus réaliste).
5) Cliquer **Générer** -> lire le tableau **Top K**.
6) Télécharger l'Excel si besoin.

Astuce : si aucun candidat ne passe, augmenter N ou relâcher les seuils.
'''
    )

    st.markdown(
        '''
### Modèles finaux et métriques (focus UCS)
**UCS est la cible principale.** Les valeurs ci-dessous viennent des CV 5-fold du projet.

**Modèles utilisés**
- **WW** : UCS = **GBR**, Slump = **GBR**
- **L01 OLD** : UCS = **ET**, Slump = **GBR**
- **L01 NEW** : UCS = **ET**, Slump = **GBR**

**Métriques UCS (R2 / RMSE kPa)**
- **WW (OLD)** : R2 ~ 0.938, RMSE ~ 168.5
- **L01 OLD** : R2 ~ 0.459, RMSE ~ 359.4
- **L01 NEW** : R2 ~ 0.859, RMSE ~ 230.4

*(Info Slump)* : RMSE ~ 26–43 mm selon le dataset.
'''
    )

    st.markdown(
        '''
### Pourquoi les métriques diffèrent entre datasets ?
- **Taille d'échantillon** : L01 OLD a moins de points utilisables, donc variance plus forte.
- **Schéma de features** : L01 NEW est plus riche (muscovite_* + ratio), ce qui aide UCS.
- **Variabilité labo / bruit** : certaines campagnes sont plus hétérogènes, RMSE plus élevé.
- **Distribution différente** : WW est plus homogène, donc R2 plus haut.
'''
    )


if __name__ == "__main__":
    main()
