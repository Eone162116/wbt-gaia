# Sorties archivées : pas de η à 0,01 dex (v0.9.1 et v0.9.2)

Ces trois fichiers sont les sorties publiées dans les commits `199d913` (v0.9.1) et `9ec7aa5` (v0.9.2). Ils ont été calculés
avec un `jointlik.py` où l'ancien bloc de constantes v0.9 réaffectait `ETA` sur 11 points (pas de 0,01 dex), au lieu des
21 points (pas de 0,005 dex) gelés le 18/10/2026 et documentés. Le défaut a été relevé par l'audit externe du 19/10/2026
(point L) et corrigé dans la v0.9.3, qui relance les trois études à l'identique (mêmes graines). Ces fichiers sont conservés
tels quels pour la traçabilité ; les sorties de référence sont celles du dossier parent.

- `stability_dr3.csv` : critères (c) 0,97 et (d) 1,72 ;
- `promotion_dr3.csv` : gain +0,65, biais maximal 0,50σ (f_in = 0,5) ;
- `power_origin_dr3.txt` : séparation 2,20σ, biais +0,81σ, influence 1,52 ; variante tronquée 2,24σ, +0,93σ, 2,06.
