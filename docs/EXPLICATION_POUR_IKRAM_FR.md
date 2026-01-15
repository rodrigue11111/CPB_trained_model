# Guide rapide pour Ikram (version simple)

## 1) A quoi sert ce projet ?
Ce projet sert a :
- predire le Slump (mm) et la resistance UCS28d (kPa)
- proposer des recettes CPB qui respectent des seuils minimaux
- exporter un fichier Excel lisible (Top_Recipes.xlsx)

## 2) Fichiers a preparer
Place les fichiers dans `data/` :
- WW : `data/WW-Optimisation.xlsx` (feuille "Feuil3 (3)")
- L01 old : `data/L01-Optimisation.xlsx` (feuille "Feuil1 (2)")
- L01 new : `data/L01-dataset.xlsx` (premiere feuille)

Colonnes minimales obligatoires :
- Binder
- E/C
- Cw_f
- Ad %
- Slump (mm)
- UCS28d (kPa)

Toutes les autres colonnes numeriques sont conservees (ex: P20 (um), P80 (um), muscovite_added, muscovite_total, etc.).

## 3) Lancer un run simple (combined)
```bash
python -m src.cli --l01 data/L01-Optimisation.xlsx --ww data/WW-Optimisation.xlsx \
  --sheet-l01 "Feuil1 (2)" --sheet-ww "Feuil3 (3)" \
  --slump-min 70 --ucs-min 900 --n-samples 50000 --out outputs/Top_Recipes.xlsx
```

## 4) Lancer un run hybride (WW et L01 separes)
```bash
python -m src.cli --fit-mode hybrid_by_tailings \
  --l01 data/L01-Optimisation.xlsx --ww data/WW-Optimisation.xlsx \
  --ww-model gbr --ww-tune false --l01-model et --l01-tune true \
  --search-mode bootstrap --n-samples 50000 --out outputs/Top_Recipes.xlsx
```

## 5) Lire le resultat
Le fichier `Top_Recipes.xlsx` contient 4 onglets :
- L01_GUL
- L01_Slag
- WW_GUL
- WW_Slag

Chaque ligne = une recette candidate.
Les colonnes `Slump_pred` et `UCS_pred` sont les predictions du modele.
Les lignes sont triees pour maximiser UCS, puis minimiser E/C et Ad %.

## 6) Comprendre les limites
- Les predictions ne remplacent pas les tests labo.
- Les datasets peuvent etre petits ou bruites.
- Il faut valider les recettes avant de les appliquer.

## 7) Verifier les colonnes (debug)
```bash
python -m src.cli --print-columns --l01 data/L01-dataset.xlsx --ww data/WW-Optimisation.xlsx
```

Cette commande affiche les colonnes detectees et celles utilisees par le modele.
