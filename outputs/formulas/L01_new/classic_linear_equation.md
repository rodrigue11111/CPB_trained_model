# Equation classique (L01 NEW)

Forme générale :
UCS = intercept + somme(coef * variable)

Intercept = -8096.2626

Termes :
- +419.0878 * Gs
- +230.7892 * Binder=20G80S
- -230.7892 * Binder=GUL
- -115.9450 * E/C
- +88.4569 * P80 (µm)
- -41.4015 * Ad %
- +36.4561 * muscovite_ratio
- +33.8206 * Cw_f
- -18.4575 * phyllosilicates (%)
- -13.1245 * P20 (µm)
- +12.0953 * muscovite_added (%)
- +10.0835 * muscovite_total (%)

Interprétation rapide :
- Un coefficient positif augmente UCS quand la variable augmente.
- Un coefficient négatif diminue UCS quand la variable augmente.

Performance test : R^2 = 0.505, RMSE = 282.84 kPa

Features retenues :
- muscovite_added (%)
- muscovite_total (%)
- phyllosilicates (%)
- P20 (µm)
- P80 (µm)
- Gs
- E/C
- Cw_f
- Ad %
- muscovite_ratio
- Binder

Avertissement : équation valable dans la plage des données L01 NEW.
