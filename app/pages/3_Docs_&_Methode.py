# -*- coding: utf-8 -*-
'''Page Documentation et m\u00e9thode Streamlit.'''

from __future__ import annotations

import streamlit as st

from app.ui.components import load_css


def main() -> None:
    st.title("Docs & M\u00e9thode")
    load_css("app/ui/styles.css")

    st.markdown(
        '''
### Pourquoi 3 datasets ?
- **WW** et **L01 OLD** viennent de campagnes diff\u00e9rentes et ont des distributions distinctes.
- **L01 NEW** a un sch\u00e9ma enrichi (muscovite_added/total + muscovite_ratio).
- Le mod\u00e8le s\u00e9lectionne dynamiquement les features disponibles, sans forcer un sch\u00e9ma unique.

### Bootstrap vs Uniform (sampling)
- **Uniform** : tirage al\u00e9atoire entre min et max observ\u00e9s.
- **Bootstrap** : tirage parmi les valeurs observ\u00e9es (plus r\u00e9aliste).

### Tuning vs Fixed Params
- **Tuning** : recherche d'hyperparam\u00e8tres (RandomizedSearchCV).
- **Fixed params** : r\u00e9utilisation des meilleurs param\u00e8tres d'un sweep pour reproduire un run.

### R\u00b2 et RMSE
- **R\u00b2** proche de 1 : bonne explication de la variance.
- **RMSE** : erreur moyenne en unit\u00e9 cible (kPa / mm).
- Dans l'app, le +/- affich\u00e9 correspond au RMSE (approx.), **pas** un intervalle statistique.

IMPORTANT : **le mod\u00e8le guide, le labo confirme**.
'''
    )

    st.markdown(
        '''
### Pipeline (sch\u00e9ma simplifi\u00e9)
```
Excel -> Nettoyage -> Features -> Entra\u00eenement -> M\u00e9triques
                             -> Optimisation -> Top_Recipes.xlsx
```
'''
    )

    st.markdown(
        '''
### Score de classement (g\u00e9n\u00e9rateur)
- **Mode contraintes min** : score = UCS_pred - 0.02 * E/C - 0.02 * Ad % (si pr\u00e9sents).
- **Mode cible** : on p\u00e9nalise l'\u00e9cart aux cibles UCS/Slump.
- Le score sert uniquement \u00e0 **ordonner** les candidats, pas \u00e0 certifier une recette.
'''
    )

    st.markdown(
        '''
### Comment utiliser le g\u00e9n\u00e9rateur de recettes
1) Choisir le **dataset** (WW / L01 OLD / L01 NEW).
2) D\u00e9finir l'objectif :
   - **Contraintes min** (Slump >= X, UCS >= Y), ou
   - **Cible + tol\u00e9rance** (ex: UCS 1200 +/- 100).
3) Indiquer **N candidats** (plus N est grand, plus on explore).
4) Choisir **Search mode** :
   - **Uniform** : tirage entre min/max.
   - **Bootstrap** : tirage dans les valeurs observ\u00e9es (plus r\u00e9aliste).
5) Cliquer **G\u00e9n\u00e9rer** -> lire le tableau **Top K**.
6) T\u00e9l\u00e9charger l'Excel si besoin.

Astuce : si aucun candidat ne passe, augmenter N ou rel\u00e2cher les seuils.
'''
    )

    st.markdown(
        '''
### Mode avanc\u00e9 : contraintes par variable
- **Preset LOT (mat\u00e9riau)** : granulom\u00e9trie et min\u00e9ralogie (P20/P80, Gs, phyllosilicates, muscovite_*).
- **Preset RECETTE (proc\u00e9d\u00e9)** : param\u00e8tres li\u00e9s au dosage (E/C, Cw_f, Ad %).
- L'id\u00e9e : si l'ing\u00e9nieur conna\u00eet son lot, il peut fixer ou borner ces variables pour proposer des recettes r\u00e9alistes.
- Les autres variables restent tir\u00e9es automatiquement (uniform ou bootstrap).
'''
    )

    st.markdown(
        '''
### Mod\u00e8les finaux et m\u00e9triques (focus UCS)
**UCS est la cible principale.** Les valeurs ci-dessous viennent des CV 5-fold du projet.

**Mod\u00e8les utilis\u00e9s**
- **GradientBoostingRegressor (GBR)** : ensemble d'arbres ajout\u00e9s s\u00e9quentiellement, chaque arbre corrige l'erreur du pr\u00e9c\u00e9dent.
- **ExtraTreesRegressor (ET)** : ensemble d'arbres tr\u00e8s al\u00e9atoires, robuste aux non-lin\u00e9arit\u00e9s.
- **Slump** : mod\u00e8le s\u00e9par\u00e9 de l'UCS, entra\u00een\u00e9 sur la cible Slump (m\u00eames familles de mod\u00e8les).

**Mod\u00e8les finaux par dataset**
- **WW** : UCS = GradientBoostingRegressor (GBR), Slump = GradientBoostingRegressor (GBR)
- **L01 OLD** : UCS = ExtraTreesRegressor (ET), Slump = GradientBoostingRegressor (GBR)
- **L01 NEW** : UCS = ExtraTreesRegressor (ET), Slump = GradientBoostingRegressor (GBR)

**M\u00e9triques UCS (R\u00b2 / RMSE kPa)**
- **WW (OLD)** : R\u00b2 ~ 0.938, RMSE ~ 168.5
- **L01 OLD** : R\u00b2 ~ 0.459, RMSE ~ 359.4
- **L01 NEW** : R\u00b2 ~ 0.859, RMSE ~ 230.4

*(Info Slump)* : RMSE ~ 26-43 mm selon le dataset.
'''
    )

    st.markdown(
        '''
### Mod\u00e8les test\u00e9s dans les sweeps (non retenus)
- **RandomForestRegressor (RF)** : solide mais pas meilleur que GBR/ET sur le compromis R\u00b2/RMSE.
- **HistGradientBoostingRegressor (HGB)** : rapide, mais r\u00e9sultats moins stables selon les seeds.
- **Support Vector Regressor (SVR)** : sensible aux hyperparam\u00e8tres et \u00e0 l'\u00e9chelle des variables.
- **ElasticNet** : mod\u00e8le lin\u00e9aire r\u00e9gularis\u00e9, sous-ajust\u00e9 pour des relations non lin\u00e9aires.

Le choix final maximise l'UCS (R\u00b2 haut, RMSE bas) tout en restant stable entre seeds.
'''
    )

    st.markdown(
        '''
### Pourquoi les m\u00e9triques diff\u00e8rent entre datasets ?
- **Taille d'\u00e9chantillon** : L01 OLD a moins de points utilisables, donc variance plus forte.
- **Sch\u00e9ma de features** : L01 NEW est plus riche (muscovite_* + ratio), ce qui aide UCS.
- **Variabilit\u00e9 labo / bruit** : certaines campagnes sont plus h\u00e9t\u00e9rog\u00e8nes, RMSE plus \u00e9lev\u00e9.
- **Distribution diff\u00e9rente** : WW est plus homog\u00e8ne, donc R\u00b2 plus haut.
'''
    )


if __name__ == "__main__":
    main()
