# Documentation technique du projet CPB

# 1. Vue d'ensemble

Ce projet automatise l'entrainement de deux modeles ML (Slump et UCS28d) et la generation de recettes CPB via un echantillonnage Monte-Carlo sous contraintes. L'objectif est de proposer des recettes filtrant un seuil minimal de Slump et de UCS, puis de fournir un export Excel avec les meilleures combinaisons.

Les donnees sont gerees par scenario :
- WW (tailings WW)
- L01_old (fichier L01-Optimisation.xlsx)
- L01_new (fichier L01-dataset.xlsx, schema different)

Pourquoi separer ces jeux ? Parce que les distributions, les colonnes disponibles et la qualite des mesures peuvent differer. Le projet applique une normalisation minimale (colonnes cibles + colonnes coeur), puis selectionne dynamiquement les features disponibles pour chaque dataset.

Resultats attendus :
- Top_Recipes.xlsx (4 onglets : L01_GUL, L01_Slag, WW_GUL, WW_Slag)
- metrics.json (qualite ML + parametres)
- best_params.json (si tuning/fixed params)
- modeles sauvegardes en joblib + metadata.json

Diagramme ASCII du flux

Excel (L01 + WW)
  |
  v
clean_dataframe + standardize_required_columns
  |
  v
Ajout Tailings + concat
  |
  +--> make_training_frames (Option A)
  |       |-> df_slump (Slump connu)
  |       |-> df_ucs   (UCS connu)
  v
infer_feature_columns
  |
  v
Pipeline ML (impute + OHE + modele)
  |
  v
CV + metrics.json
  |
  v
Optimisation Monte-Carlo
  |
  v
Top_Recipes.xlsx + modeles joblib

# 2. Architecture du repo (carte des fichiers)

Arborescence cle :

cpb_optimization/
  src/
    cli.py                # CLI principale (entrainement + optimisation)
    schema.py             # Nettoyage minimal + normalisation des colonnes requises
    io_data.py            # Lecture Excel + validation
    features.py           # Selection dynamique features + preprocess
    train.py              # Pipelines ML, CV, tuning, hybrid
    optimize.py           # Echantillonnage Monte-Carlo + export
    sweep.py              # Sweep multi-datasets + resumes CSV
    compare_runs.py       # Comparaison de runs (legacy)
    config.py             # Constantes et defaults
  scripts/
    retrain_best_models.py # Retrain final a partir des sweeps
  tests/
    test_io_features.py
    test_pipeline.py
    test_hybrid_metrics.py
  outputs/                # Artefacts (ignore git)
  docs/
    EXPLICATION_PROJET_FR.md
    EXPLICATION_POUR_IKRAM_FR.md

# 3. Pipeline de donnees

## Lecture Excel
- src.io_data.read_excel_file lit une feuille via openpyxl.
- Si la feuille n'est pas fournie, la premiere feuille est prise (auto-detection).
- La colonne Tailings est ajoutee manuellement ("L01" ou "WW").

## Nettoyage minimal / normalisation
- src.schema.clean_dataframe
  - strip des headers
  - supprime les colonnes vides
  - convertit en numerique si >= 50% convertible
  - garde Binder/Tailings en string
  - ajoute muscovite_ratio si possible (muscovite_added / muscovite_total)

- src.schema.standardize_required_columns
  - renomme uniquement les colonnes obligatoires vers les noms canoniques :
    - UCS28d (kPa)   (alias possibles: UCS28d(kPa), UCS28d\n(kPa))
    - Slump (mm)     (alias possibles: Slump(mm), Slump\n(mm))
    - Binder         (alias: binder)
    - E/C            (alias: E/C )
    - Cw_f           (alias: Cw_f )
    - Ad %           (alias: Ad%, Ad  %)
  - toute autre colonne est conservee telle quelle

Si une colonne obligatoire manque, une ValueError est levee avec la liste des colonnes disponibles.

## Feature engineering specifique NEW
Si muscovite_added et muscovite_total sont disponibles (apres nettoyage), on cree :
- muscovite_ratio = muscovite_added / muscovite_total (si total > 0)

Les colonnes d'origine restent intactes.

# 4. Construction des features et preparation ML

## Selection dynamique des features
On ne force pas une liste fixe de colonnes. Chaque dataset utilise ses propres features disponibles :
- Categorielles : Binder + Tailings (si >1 valeur unique)
- Numeriques : toutes les colonnes numeriques sauf targets
- Colonnes constantes ou vides: retirees

Exemples typiques (non exhaustifs) :
- OLD: muscovite (%), phyllosilicates (%), P80 (%), D80 (um), Gs, E/C, Cw_f, Ad %
- NEW: P20 (um), P80 (um), muscovite_added (%), muscovite_total (%), muscovite_ratio, E/C, Cw_f, Ad %

## Encodage et imputation
- Categorielles: imputation most_frequent + OneHotEncoder
- Numeriques: imputation median
- Pour SVR / ElasticNet: StandardScaler sur features numeriques

## Option A (pas de fuite)
On n'utilise jamais Slump pour predire UCS, ni l'inverse.
- df_slump = lignes ou Slump (mm) est connu
- df_ucs   = lignes ou UCS28d (kPa) est connu

## Unites (rappel)
- UCS: kPa
- Slump: mm
- granulometrie: um (micro-metres)
- pourcentages: %

# 5. Modeles et entrainement

## Modeles supportes
- GBR: GradientBoostingRegressor
- HGB: HistGradientBoostingRegressor
- RF: RandomForestRegressor
- ET: ExtraTreesRegressor
- SVR: Support Vector Regression (RBF)
- ENet: ElasticNet

Differences rapides entre modeles
- GBR: boosting sequentiel de petits arbres, capte des non linearites fines.
- HGB: variante plus rapide du boosting (hist bins), stable sur plus de donnees.
- RF: bagging d'arbres, reduit la variance, robuste aux outliers moderes.
- ET: arbres tres randomises, variance plus faible, parfois meilleur que RF.
- SVR: noyau RBF, puissant mais sensible aux hyperparametres et a l'echelle.
- ENet: regression lineaire regularisee (L1+L2), simple mais peut sous-ajuster.

Pourquoi les arbres/boosting marchent souvent mieux ?
- Relations non lineaires
- Robustesse aux interactions entre variables
- Peu sensibles aux echelles (sauf SVR/ENet qui sont scales)

## Transformations et outliers (UCS)
- transform UCS = none | log (log1p / expm1)
- outliers = none | iqr | zscore
  - iqr: garde [Q1-1.5*IQR, Q3+1.5*IQR]
  - zscore: garde |z| <= 3

## CV et metriques
- KFold 5 plis (shuffle=True, seed fixe)
- Metriques: RMSE (sqrt(MSE)) et R2
- Rapports globaux + par groupe Tailings et Binder

## Seed
- random.seed + np.random.seed
- controle aussi la recherche Monte-Carlo
- assure la reproductibilite

Extrait (selection dynamique + pipeline)
```python
categorical_cols, numeric_cols = infer_feature_columns(
    df, target_cols=[TARGET_SLUMP, TARGET_UCS]
)
pipe = build_pipeline(
    "gbr",
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    random_state=seed,
)
```

# 6. Les fit_mode

- combined
  - un seul modele Slump et un seul modele UCS sur l'ensemble du dataset
- by_tailings
  - un modele WW et un modele L01 mais avec la meme config globale
- hybrid_by_tailings
  - un modele WW et un modele L01, chacun avec sa propre config
  - transform/outliers/tuning par tailings
  - utile quand WW et L01 ont des distributions differentes

# 7. La CLI (src/cli.py)

Commande de base
```bash
python -m src.cli --l01 data/L01-Optimisation.xlsx --ww data/WW-Optimisation.xlsx \
  --sheet-l01 "Feuil1 (2)" --sheet-ww "Feuil3 (3)" \
  --slump-min 70 --ucs-min 900 --n-samples 50000 --out outputs/Top_Recipes.xlsx
```

Arguments importants
- --fit-mode combined|by_tailings|hybrid_by_tailings
- --model gbr|rf|et|hgb|svr|enet (mode combined)
- --ww-model / --l01-model (mode hybrid)
- --ww-ucs-transform / --l01-ucs-transform (none|log)
- --ww-ucs-outliers / --l01-ucs-outliers (none|iqr|zscore)
- --ww-tune / --l01-tune, --ww-slump-tune / --l01-slump-tune
- --search-mode uniform|bootstrap
- --seed pour reproductibilite
- --run-id + --out-dir pour organiser les sorties
- --save-models true|false et --models-dir pour joblib
- --print-columns pour debug (affiche colonnes puis exit)

Tuning vs fixed params
- Tuning: RandomizedSearchCV, enregistre best_params.json
- Fixed params: JSON passe via --*_fixed-params, puis set_params
  - Le tuning est force OFF pour ce modele
  - Garantit une reproduction exacte d'un meilleur run

Sauvegarde des modeles
- outputs/models/<run-id>/
  - ww_ucs.joblib, ww_slump.joblib, l01_ucs.joblib, l01_slump.joblib
  - metrics.json
  - best_params.json
  - metadata.json

metadata.json contient :
- run_id, seed, fit_mode
- chemins des datasets + feuilles
- meta models (nom, params, transform/outliers)
- features utilisees

Reproduire un run :
- reprendre la commande stockee dans `outputs/<...>/runs/<run_id>/command.txt`
- conserver le meme seed et, si present, les fixed params

# 8. Optimisation / generation de recettes (src/optimize.py)

## Sampling
- uniform: tirage uniforme entre min/max observes
- bootstrap: tirage aleatoire dans les valeurs observees
- si un sous-groupe (Tailings,Binder) a <10 points, on prend les bornes du Tailings global

## Filtrage
- conserve uniquement si:
  - Slump_pred >= slump_min
  - UCS_pred >= ucs_min

## Tri final
- tri par UCS_pred desc, puis E/C asc, puis Ad % asc
- conserve top N (default 25)

## Export
- Top_Recipes.xlsx avec 4 onglets (L01_GUL, L01_Slag, WW_GUL, WW_Slag)
- Colonnes: Tailings + Binder + features numeriques + Slump_pred + UCS_pred

Extrait (filtrage + tri)
```python
filtered = candidates[
    (candidates["Slump_pred"] >= slump_min)
    & (candidates["UCS_pred"] >= ucs_min)
]
filtered = filtered.sort_values(
    by=["UCS_pred", "E/C", "Ad %"],
    ascending=[False, True, True],
)
```

# 9. Sweep (src/sweep.py)

Un sweep lance de nombreux runs avec des configs differentes, puis construit des tableaux de synthese.

Points cle :
- run_id stable (dataset + seed + configs)
- budget temps "souple" : on avertit quand la limite est depassee, mais on continue
- resume: si metrics.json existe, on saute le run
- logs par run: command.txt + run.log + status.json
- fichiers de synthese:
  - sweep_runs.csv (1 ligne par run)
  - sweep_configs.csv (moyennes/ecarts type par config)

Si "Aucun run valide", cela signifie que les metrics.json attendus ne sont pas presents ou que tous les runs ont echoue.

# 10. Retrain FINAL (scripts/retrain_best_models.py)

Ce script relance un entrainement final a partir des meilleurs sweeps :
- Cherche la meilleure config via final_score
- Charge best_params.json si disponible
- Appelle la CLI avec --*_fixed-params pour reproduire exactement le meilleur run

Pourquoi c'est reproductible ?
- Les params fixes repliquent exactement les hyperparametres trouves au sweep
- Le seed est fixe

# 11. Guide d'utilisation (pas-a-pas)

1) Verifier les colonnes du nouveau dataset
```bash
python -m src.cli --print-columns --l01 data/L01-dataset.xlsx --ww data/WW-Optimisation.xlsx
```

2) Entrainement simple (combined)
```bash
python -m src.cli --l01 data/L01-Optimisation.xlsx --ww data/WW-Optimisation.xlsx \
  --slump-min 70 --ucs-min 900 --n-samples 20000 --out outputs/Top_Recipes.xlsx
```

3) Entrainement hybride par tailings
```bash
python -m src.cli --fit-mode hybrid_by_tailings --l01 data/L01-Optimisation.xlsx \
  --ww data/WW-Optimisation.xlsx --search-mode bootstrap --n-samples 50000 \
  --ww-model gbr --ww-tune false --l01-model et --l01-tune true
```

4) Lancer un sweep 3h sur dataset NEW
```bash
python -m src.sweep --only-dataset new --out-dir outputs/sweep_new_3h \
  --time-budget-min 180 --resume --n-samples 15000
```

5) Retrain final depuis un sweep
```bash
python scripts/retrain_best_models.py --sweep-new outputs/sweep_new_3h \
  --seed 42 --n-samples 50000 --out-dir outputs/final_models
```

# 12. Limites, risques et ameliorations

Limites / risques :
- Les performances L01_old restent modestes: le dataset est petit et plus bruite.
- Les predictions sont des estimations, pas des garanties en laboratoire.
- Les features utiles peuvent manquer (teneur en eau, temperature, cure, etc).

Idees d'ameliorations :
- Collecter plus d'essais L01 et WW
- Ajouter des variables de contexte (temperature, cure, temps)
- Tester des methodes d'ensembling ou des modeles bayesiens
- Ameliorer la calibration des seuils (slump_min / ucs_min)

Ambiguite a verifier :
- Certaines colonnes du dataset NEW sont nommees avec des symboles (ex: "P20 (um)").
  Le code ne renomme pas ces colonnes, donc il suffit que ces colonnes soient numeriques.
  Pour verifier, utiliser `--print-columns` et inspecter la liste.
