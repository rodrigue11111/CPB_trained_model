# Formule globale (Spline + ElasticNet)

Forme generale :
UCS = b0 + somme(coeff * bases_splines) + effets du liant (Binder).

Meilleurs parametres :
- alpha = 0.1
- l1_ratio = 0.9

Performance test : R^2 = 0.477, RMSE = 290.79 kPa

Top 30 termes (coefficients les plus influents) :
- intercept: 912.0470
- Ad %_sp_4: 295.3674
- Ad %_sp_6: -234.3527
- Binder_20G80S: 225.4961
- Binder_GUL: -225.4959
- Ad %_sp_2: -199.2906
- Cw_f_sp_6: 194.9465
- E/C_sp_1: 192.9975
- Ad %_sp_3: 171.7313
- E/C_sp_5: -139.5598
- Cw_f_sp_2: -127.4740
- E/C_sp_6: -72.1972
- P80 (µm)_sp_6: 57.9174
- P80 (µm)_sp_2: -56.6422
- Gs_sp_6: 52.6016
- phyllosilicates (%)_sp_1: 51.7102
- muscovite_ratio_sp_1: 51.2925
- Cw_f_sp_1: -50.4039
- muscovite_total (%)_sp_1: 50.3551
- Ad %_sp_1: 45.3910
- E/C_sp_0: 42.5947
- muscovite_total (%)_sp_3: -42.1700
- Cw_f_sp_3: -39.3217
- Ad %_sp_5: -38.5601
- Gs_sp_5: -38.1625
- P20 (µm)_sp_6: -38.1175
- muscovite_added (%)_sp_1: 37.2994
- P20 (µm)_sp_5: 35.5095
- muscovite_added (%)_sp_3: -34.4978
- muscovite_ratio_sp_3: -29.4767

Limites: valable dans la plage des donnees L01 NEW. Hors plage, la formule extrapole.