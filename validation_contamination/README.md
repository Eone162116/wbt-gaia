# Modèle de contamination par compagnons cachés (16/10/2026)

Question : la marche résiduelle observée sur DR3 (S ≈ +0,05 à +0,07) peut-elle venir de compagnons internes non détectés, sous gravité newtonienne ?

Méthode (`contamination_model.py`, à lancer depuis la racine après `run_wbt.py dr3 build`) :
- géométrie réelle de l'échantillon principal DR3, avec une vitesse newtonienne pour vérité ;
- pour chaque composante, un compagnon interne avec une probabilité f_in, une période log-normale (Raghavan et al. 2010), q uniforme sur [0,05 ; 1] et une stabilité a_in < 0,2 × la séparation de la binaire large ;
- effets modélisés : masse cachée et vitesse moyenne du photocentre sur les 5,5 ans de DR4 ;
- cribles : époques (`pgrid.npy`, mesuré par injection sur l'échantillonnage réel de la pré-publication DR4, voir `epoch_grid.py`) et image, en scénario pessimiste (résolution du catalogue DR3) ou optimiste (diffraction à 0,15″).

Résultats (`contamination_results.csv`, `fig_contamination_bias.png`), biais sur S :

| Crible | f_in = 0,3 | f_in = 0,5 |
|---|---|---|
| aucun | +0,041 | +0,082 |
| époques seules | +0,063 | +0,111 |
| époques + image (pessimiste) | +0,024 | +0,043 |
| époques + image (optimiste) | +0,007 | +0,014 |

Ce qu'il faut en retenir :
- une contamination réaliste suffit à produire le résidu DR3 sous Newton ;
- le crible par époques seul aggrave le biais : il retire les compagnons proches, qui relèvent ṽ à toutes les séparations, et garde les lointains, qui créent la marche ;
- c'est la détection image entre 30 et 300 au qui décide ;
- le test différentiel en distance (S loin − S proche) n'a pas de puissance à N ≈ 1000 : l'écart-type est d'environ 0,15–0,22, pour des effets inférieurs à 0,03.

Limites : distribution de périodes établie pour des étoiles de type solaire (les naines M ont des compagnons plus serrés) ; modèle d'image simplifié ; crible RV non modélisé ; relation masse–luminosité approchée.
