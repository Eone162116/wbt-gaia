# Sorties archivées : entrées du mode rapide (v0.9.3)

Ces fichiers sont les sorties publiées dans le commit `3e30878` (v0.9.3), calculées avec le pas de η corrigé (0,005 dex) mais à
partir d'entrées aveugles produites en mode rapide (WBT_QUICK=1) : excentricité α = (2,00 ; 0,25). La décision de non-promotion
du 18/10/2026 a été prise sur ces valeurs. Les sorties du dossier parent reprennent les mêmes études sur les entrées de la
répétition DR3 complète du 19/10/2026 (mode complet : α = (2,00 ; 0,50)). Les deux jeux d'entrées se distinguent par l'empreinte
de `eccfit_dr3.csv` dans les fichiers `*_provenance.json` (mode rapide b0a464bb…, mode complet 558a3fbf…).

| Étude | Entrées du mode rapide (ici) | Entrées du mode complet (dossier parent) |
|---|---|---|
| Stabilité (c) / (d) | 0,97 / 1,72 | 0,97 / 1,56 |
| Promotion : gain / biais maximal | +0,64 / 0,50σ | +0,51 / 0,52σ |
| Origine : séparation / biais f_in 0,5 / influence | 2,21σ / +0,81σ / 1,63 | 2,05σ / +0,26σ / 1,51 |

Les critères (b) et (d) restent non satisfaits, avec de faibles marges, du même ordre que l'incertitude Monte-Carlo
(20 à 200 jeux selon l'étude). La décision était close avant ce recalcul et n'en dépend pas.
