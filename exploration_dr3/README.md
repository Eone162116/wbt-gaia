# WBT — test des binaires larges (MOND vs Newton) sur Gaia DR3

Pipeline headless de reproduction de Cookson, Banik, El-Badry et al. 2026 (MNRAS, arXiv:2602.24035), conçu pour un multivers de spécifications et réutilisable tel quel sur Gaia DR4. État au 16/10/2026.

## Exécution (≈ 20 min, dont 15 de téléchargement)

```bash
pip install numpy pandas scipy matplotlib requests
M=3 R=0 python fetch_gaia.py & M=3 R=1 python fetch_gaia.py & M=3 R=2 python fetch_gaia.py & wait
python build_pairs.py        # 1 426 271 étoiles -> 133 213 liens -> 58 806 paires mutuelles -> K.pkl
python run_multiverse.py     # 144 spécifications, CSV + 2 figures
```

Pièges du TAP ESA en anonyme : jobs asynchrones tués à 60 s ; requêtes synchrones tronquées silencieusement à 50 000 lignes. D'où 40 tranches de parallaxe à volume égal (max 46 497 lignes) ; le total est vérifié contre un COUNT(*) serveur.

## Chaîne

1. Catalogue large : parallaxe > 6 mas, S/N > 10 (sert au dégroupage). Étoiles principales : S/N > 40 (≈ catalogue C25 de Cookson).
2. Compagnons dans 100 kau : parallaxes compatibles à 3σ, |Δμ| < 0,44 ϖ^1,5 θ^-0,5 + 2σ (El-Badry & Rix 2018).
3. Paires mutuelles ; dégroupage = chaque étoile n'a qu'un seul compagnon dans 100 kau.
4. Distance commune pondérée ; masses par relation M_G (Cookson éq. 12) ; v_sky avec correction sphérique (vitesses 3D, RV systémique commune) ; incertitude sur ṽ propagée.
5. Coupes : 1–30 kau, d < 130 pc, RUWE < 1,25, deux RV et ΔRV < 10 km/s, Δv_sky < 10 km/s, ṽ < 2,5, σ(ṽ) < 0,1, parallélogramme HR, option ipd_frac_multi_peak = 0.
6. Médiane de ṽ dans 6 bins de r_sky/r_M ; χ² Newton (plat) vs MOND (Cookson éq. 7), normalisation libre ; préférence = exp(Δχ²/2).

## Résultats

| Variante | Paires | Newton/MOND | Cookson 2026 |
|---|---|---|---|
| V0 checklist complète | 1112 | 106 | 1421 / 548 |
| V1 + ipd = 0, ṽ < 1,5 | 886 | 62 | 1216 / 1501 |
| V2 sans correction sphérique | 1112 | 74 | effet +0,15 annoncé |
| V3 sans filtre HR | 2797 | 1,0e5 | MOND-like annoncé |
| V4 150 pc | 1479 | 102 | MOND 12× |
| V5 150 pc + ipd = 0 | 1193 | 10 | Newton 88× |
| V6 ṽ < 5 | 1112 | 106 | — (aucune paire entre 2,5 et 5 après les autres coupes) |
| V7 sans dégroupage | 1245 | **0,11 (MOND 9×)** | — |
| V8 dégroupage sur S/N > 40 | 1121 | 74 | — |

Multivers (144 spécifications = dégroupage × filtre HR × correction sphérique × ipd × distance × RUWE) : 54 favorisent MOND. Sans dégroupage, 46 sur 48 (96 %). Avec dégroupage, 8 sur 96, toutes avec bande HR lâche (1,2 mag) et sans filtre ipd, la plupart sans correction sphérique. Médiane log10(Newton/MOND) : −2,3 sans dégroupage, +1,8 avec. Distance et seuil RUWE pèsent peu dans cette implémentation.

Lecture : la conclusion suit le contrôle de la contamination par compagnons cachés. Chaque relâchement de ce contrôle pousse vers MOND. C'est cohérent avec la thèse de Cookson et al., mais pas avec leur résultat à 150 pc : avec un catalogue de dégroupage plus inclusif (S/N > 10 au lieu de > 40), l'effet distance disparaît ici.

## Écarts connus avec Cookson 2026 et limites

- Intercept du parallélogramme HR non publié : bande centrée sur la médiane de M_G − 3,3(BP−RP). Elle garde 56 % des étoiles de la boîte, probablement trop serrée. C'est la cause principale probable de l'écart 1112 vs 1421.
- Critères d'appariement El-Badry, pas ceux du code bynary ; masses sans FLAME ; pas de bin r/r_M < 0,1 avec s ≥ 1 kau.
- Statistique fragile : bins de 49 à 450 paires, erreur MAD/√N. Dans les 8 spécifications dégroupées pro-MOND, χ² ≈ 15–22 pour 4 ddl : aucun des deux modèles n'ajuste, la « préférence » compare deux mauvais ajustements. Il faut ajouter un test d'adéquation et un bootstrap.
- Pipeline de travail, pas un résultat publiable. Objectif suivant : réduire l'écart à Cookson (HR, critères de paires), puis figer les choix dans le pré-enregistrement avant DR4.

## Test d'injection-récupération et calibration (16/10/2026)

`python injection_recovery.py` (≈ 3 min). Les masses, distances, séparations et incertitudes des 1112 paires réelles sont conservées. Leurs vitesses sont remplacées par des vitesses simulées : orbites képlériennes, excentricité thermique, orientation et phase aléatoires. Le boost MOND est appliqué au rayon 3D tiré. Une fraction de compagnons cachés est ajoutée : biais de masse, plus recul du photocentre pour a_in > 10 au, la plage 0,3–10 au étant supposée éliminée par RUWE. Le bruit réel et les mêmes coupes s'appliquent ensuite. 300 réalisations par scénario.

Statistique ajoutée, plus robuste que le χ² binné : la marche S = médiane(ṽ | r/rM > 1) / médiane(ṽ | r/rM < 0,3) − 1.

| Vérité | Compagnons | S médian [IC95] | Part pro-MOND | Pro-MOND > 10× |
|---|---|---|---|---|
| Newton | 0 % | +0,005 [−0,12 ; +0,12] | 7 % | 2 % |
| MOND | 0 % | +0,150 [−0,01 ; +0,29] | 86 % | 72 % |
| Newton | 15 % | +0,002 | 7 % | 3 % |
| Newton | 30 % | +0,004 | 9 % | 2 % |
| Newton | 50 % | +0,009 | 11 % | 4 % |
| MOND | 30 % | +0,138 | 83 % | 70 % |

Résultats :

- **Le pipeline fonctionne.** Il retrouve une marche MOND injectée (S ≈ +0,15) et ne crée pas de marche à partir de Newton.
- **Sa puissance est limitée.** À N ≈ 1100, σ(S) ≈ 0,064. Le taux de faux positifs pro-MOND est de 7 % et la puissance de 86 %.
- **Données réelles.** S = +0,002, IC95 bootstrap [−0,14 ; +0,20]. Une valeur aussi basse n'arrive que dans 3,7 % des univers MOND. log10(Newton/MOND) = 2,0, valeur atteinte ou dépassée dans 2,7 % des univers MOND. Le rejet calibré de MOND est donc d'environ 2σ, et non « 106× ». C'est cohérent avec la tension MOND de 2,6σ que Cookson et al. rapportent eux-mêmes (leur tableau 3), derrière leur 1500×.
- **Adéquation.** Newton : χ² = 2,7 pour 4 ddl, p = 0,61. MOND : χ² = 12,1, p = 0,017.
- **Compagnons non résolus.** Même à 50 % de contamination, avec a_in jusqu'à 3000 au, la marche reste à S ≤ +0,03. Ils doublent ou triplent le taux de faux positifs mais n'imitent pas une marche MOND complète. La bascule observée sans dégroupage vient donc de multiples résolus et de groupes co-mouvants, que ce modèle jouet ne simule pas.
- **Dimensionnement pour DR4.** Pour séparer Newton et MOND à 3σ sur S, il faut environ 2000 paires propres. À 5σ, il en faut environ 5000.

Limites : rayon 3D tiré sans pondération par la distribution des demi-grands axes ; boost MOND appliqué à des orbites képlériennes ; modèle de contamination simplifié.

## Fichiers

- `fetch_gaia.py`, `build_pairs.py`, `wbt_pipeline.py`, `run_multiverse.py`, `injection_recovery.py` : code.
- `multiverse_results.csv`, `multiverse_factor_effects.csv`, `injection_results.csv`, `injection_summary.csv`, `bootstrap_real.csv` : sorties.
- `fig_specification_curve.png`, `fig_baseline_vs_nodegroup.png`, `fig_injection_recovery.png` : figures.
