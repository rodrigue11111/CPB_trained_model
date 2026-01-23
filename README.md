# Optimisation CPB

Pipeline pour entrainer les modeles Slump/UCS et optimiser des recettes CPB
par echantillonnage Monte-Carlo sous contraintes.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Verrouillage des versions (reproductibilite)

- `requirements.txt` est utilise par Streamlit Cloud.
- `requirements-lock.txt` fige toutes les versions pour reproduire un environnement local.

Installation locale identique :
```bash
pip install -r requirements-lock.txt
```

## Donnees

Placez les fichiers Excel dans le dossier `data/` :
- `data/L01-Optimisation.xlsx`
- `data/WW-Optimisation.xlsx`

## Execution

```bash
python -m src.cli --l01 data/L01-Optimisation.xlsx --ww data/WW-Optimisation.xlsx \
  --sheet-l01 "Feuil1 (2)" --sheet-ww "Feuil3 (3)" \
  --slump-min 70 --ucs-min 900 --n-samples 50000 --out outputs/Top_Recipes.xlsx
```

Exemple hybrid_by_tailings (modeles differents par tailings) :
```bash
python -m src.cli --l01 data/L01-Optimisation.xlsx --ww data/WW-Optimisation.xlsx \
  --fit-mode hybrid_by_tailings \
  --ww-model rf --l01-model gbr \
  --ww-ucs-transform log --l01-ucs-transform none \
  --ww-ucs-outliers iqr --l01-ucs-outliers none \
  --ww-slump-model gbr --l01-slump-model rf \
  --out outputs/Top_Recipes.xlsx
```

Option : `--search-mode` accepte `uniform` (min/max) ou `bootstrap` (tirage
sur les valeurs observees).

Si vos fichiers sont dans `Ikram_documents`, passez les chemins complets
a la place de `data/`.

Notes :
- Le modele Slump est entraine uniquement sur les lignes avec `Slump (mm)` renseigne.
- Le modele UCS est entraine uniquement sur les lignes avec `UCS28d (kPa)` renseigne.
- Aucun target n'est utilise comme feature pour l'autre (pas de fuite).
- Le Monte-Carlo genere `N` recettes par groupe Tailings/Binder
  (L01_GUL, L01_Slag, WW_GUL, WW_Slag) dans les bornes observees.

## Sweep

```bash
python -m src.sweep --out-dir outputs/sweeps
```

Par defaut, le sweep lance les scenarios "old" et "new" (L01-dataset en feuille 1).
Pour forcer uniquement le dataset "new", ajoutez `--only-dataset new`.

Exemple sweep 5 heures (grille limitee, multi-datasets) :
```bash
python -m src.sweep --out-dir outputs/sweep_5h --time-budget-min 300 --resume \
  --n-samples 15000 --slump-min 70 --ucs-min 900 \
  --seeds "42,123" --search-modes "bootstrap" \
  --ww-ucs-models "gbr,hgb" --ww-ucs-transforms "none" --ww-ucs-outliers "none" \
  --ww-tune-options "false" \
  --l01-ucs-models "rf,et,hgb" --l01-ucs-transforms "none,log" \
  --l01-ucs-outliers "none,iqr" --l01-tune-options "true" \
  --ww-slump-models "gbr" --ww-slump-tune-options "false" \
  --l01-slump-models "gbr" --l01-slump-tune-options "true"
```

## Comparer les runs

```bash
python -m src.compare_runs --out-dir outputs
```

## Documentation

- docs/EXPLICATION_PROJET_FR.md
- docs/EXPLICATION_POUR_IKRAM_FR.md

## Tests

```bash
pytest
```

## Application Streamlit

Lancement local :
```bash
streamlit run streamlit_app.py
```

Mode Avance (Generateur de recettes) :
- Activez le toggle "Mode avance" pour fixer ou borner certaines variables numeriques.
- Deux presets existent : LOT (materiau) et RECETTE (procede).

Modeles attendus (par defaut) :
- outputs/final_models/FINAL_old_best/
- outputs/final_models/FINAL_new_best/

Si les modeles ne sont pas trouves, l'app affichera un message clair.

## Visualiser les relations UCS

1) Generer les sorties d'interpretabilite (une seule fois) :
```bash
python scripts/explain_ucs_relationships.py --models-dir outputs/final_models/FINAL_new_best --dataset-xlsx data/L01-dataset.xlsx --out-dir outputs/interpretability/FINAL_new_best
```

2) Ouvrir la page "Relations (UCS)" dans l'app Streamlit et choisir le dossier.

## Deploiement Streamlit Cloud (pas a pas)

1) Pousser le repo sur GitHub (inclure streamlit_app.py et le dossier app/).
2) Dans Streamlit Cloud, creer une nouvelle app en pointant sur le repo.
3) Definir le fichier d'entree: streamlit_app.py
4) Gestion des modeles :
   - Option A (simple): commit les dossiers FINAL_* dans outputs/final_models
     (si la taille est raisonnable), ou ajuster .gitignore.
   - Option B (propre): publier un zip via GitHub Release et ajouter dans
     .streamlit/secrets.toml la cle MODEL_DOWNLOAD_URL.

Exemple secrets.toml (non versionne) :
```
MODEL_DOWNLOAD_URL = "https://.../final_models_bundle.zip"
```

## Packaging des modeles (optionnel)

Pour preparer un zip des modeles finaux (Release GitHub) :
```bash
python scripts/package_models.py
```
