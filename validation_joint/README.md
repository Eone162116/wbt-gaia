# Vraisemblance conjointe (ṽ, ψ) : étude de puissance (18/10/2026)

Simulation seulement : aucune vitesse observée n'est utilisée. La géométrie, le bruit et les coupes sont ceux de l'échantillon principal DR3. β est fixé aux valeurs mesurées sur les séparations ; la vérité est α = (2,0 ; 0,25).

Test conjoint : rapport de vraisemblance Λ entre MOND et Newton, calculé sur des gabarits 2D (log ṽ, ψ) par zone de x. Trois nuisances sont profilées : α dans chaque zone et une échelle globale η de ṽ, commune aux deux zones (±0,05 dex), qui absorbe les erreurs de masse uniformes.

Résultats (`joint_power_dr3.csv`), probabilité d'un verdict tranché à 3σ :

| Cas | Test conjoint : MOND rejeté si Newton vrai | Test conjoint : Newton rejeté si MOND vrai | Test S : MOND rejeté si Newton vrai | Test S : Newton rejeté si MOND vrai |
|---|---|---|---|---|
| Taille DR3 (725 paires dans les deux zones) | 0,06 | 0,39 | 0,00 | 0,05 |
| Taille doublée (≈ 2400 paires au total) | 0,21 | 0,57 | 0,09 | 0,30 |

À taille doublée, le test conjoint gagne 0,12 et 0,27 sur S. Le critère de promotion (au moins +0,10) est donc atteint en puissance.

Tests de robustesse : effet sur Λ sous Newton, en écarts-types de Λ.
- **Biais de masse uniforme de +20 %** : environ +0,5σ vers MOND, malgré η. Le reliquat vient de la coupe ṽ < 2,5 et du rapport signal/bruit.
- **Compagnons cachés non modélisés** (30 % des paires, coups de 0,03 à 1 km/s) : environ +1,1σ vers MOND.

Le test conjoint est donc plus puissant, mais plus dépendant du modèle. Sa promotion exige que la contamination entre dans les gabarits (bande de f_in mesurée), et qu'un test de robustesse montre un biais résiduel inférieur à 0,5σ.

Note : une première version sans η donnait 4,8σ de séparation. Ce chiffre était artificiel, car il exploitait la normalisation absolue de ṽ, donc la calibration des masses.

## v0.9.1 : stabilité du test conjoint lissé (18/10/2026)

Simulation seulement (`stability_study.py`, résultats dans `stability_dr3.csv`). Vingt jeux simulés à la géométrie DR3 (10 Newton central, 10 MOND central), cinq tirages indépendants de gabarits (R = 90). Critère (c), dispersion de Λ due aux gabarits, moyenne quadratique : 0,97, sous le seuil de 1,0. Critère (d), influence maximale d'une paire, 90e centile : 1,72, au-dessus du seuil de 1,5. Étude de promotion (`promotion_dr3.csv`, gabarits R = 30, à titre d'information) : gain de puissance +0,65 à ≈ 2400 paires, mais biais de 0,50σ pour f_in = 0,5, au seuil. La v0.9.1 n'est donc pas promue : S reste le test principal, et le test conjoint une analyse secondaire.

## v0.9.2 : origine de la puissance du test conjoint (18/10/2026)

Simulation seulement (`power_origin.py`, sortie `power_origin_dr3.txt`). Λ est décomposé paire par paire au point profilé de chaque hypothèse. Sur une séparation attendue de 10,5 unités de Λ entre MOND et Newton, 5,0 viennent des paires de la zone haute avec ṽ entre 1 et √2, 2,8 de la zone basse (ancrage de η et de la contamination), et presque rien des paires au-delà de √2. Une variante tronquée à ṽ < √2 garde la même séparation (2,24σ contre 2,20σ), mais sa sensibilité à la contamination hors bande (+0,93σ contre +0,81σ) et l'influence d'une paire (2,06 contre 1,52) sont plus fortes. La puissance du test conjoint est donc celle de distinguer MOND d'un excès de compagnons cachés dans la même région de ṽ. C'est pourquoi la question de sa promotion est close.

## v0.9.3 : correction du pas de η, provenance (19/10/2026)

L'audit externe du 19/10/2026 a relevé que `jointlik.py` réaffectait `ETA` sur 11 points (pas de 0,01 dex), au lieu des 21 points (pas de 0,005 dex) gelés le 18/10. Le code est corrigé et les trois études ont été relancées avec les mêmes graines. Les sorties précédentes sont dans `archive_eta001/`. Stabilité : (c) 0,97 et (d) 1,72, inchangés. Promotion : gain +0,64, biais maximal 0,50σ. Origine de la puissance : séparation 2,21σ, dont 5,1 unités sur 10,5 en zone haute pour ṽ entre 1 et √2 ; biais +0,81σ ; influence 1,63 ; variante tronquée 2,25σ, +0,94σ, 2,04. Chaque sortie a désormais un fichier `*_provenance.json`.

Deux précisions de portée. D'abord, aucune vitesse observée n'entre dans la statistique des jeux simulés, mais la géométrie simulée provient d'une sélection qui utilise des coupes cinématiques. Ensuite, la décomposition de Λ par tranche est descriptive et conditionnelle au profilage : elle ne mesure pas la puissance qui survivrait à la suppression d'une tranche. Les tailles Monte-Carlo sont faibles (20 à 40 jeux par hypothèse), si bien que des écarts de quelques centièmes aux seuils ne sont pas significatifs. `joint_power.py` est une étude historique dont les puissances ne se transfèrent pas au test conjoint lissé.
