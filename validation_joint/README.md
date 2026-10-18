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
