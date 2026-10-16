# wbt-gaia — test des binaires larges (Newton vs MOND), pré-enregistré et aveuglé pour Gaia DR4

v0.3 (16/10/2026). Auteur : Enguerrand Michon, chercheur indépendant (Lyon).

Ce dépôt teste si les binaires larges du voisinage solaire présentent, au-delà du rayon MOND, l'excès de vitesse orbitale prédit par MOND avec le champ externe galactique. L'analyse sera pré-enregistrée avant la sortie de Gaia DR4 (2 décembre 2026) et exécutée à l'aveugle. Un seul point d'entrée, `run_wbt.py`, et un seul fichier de choix, `config_prereg.json`, figé au dépôt du pré-enregistrement.

## Installation

```bash
python -m venv .venv
.venv/bin/activate            # Windows PowerShell : .\.venv\Scripts\Activate.ps1
pip install numpy pandas scipy matplotlib requests
```

La clé d'aveuglement est créée automatiquement dans `~/.wbt_blind_key`, hors du dépôt. On peut aussi l'indiquer avec la variable `WBT_BLIND_KEY`. Elle n'est jamais affichée ni versionnée.

## Procédure du 2 décembre 2026

1. **Essai à sec.** `SELECT TOP 1 source_id FROM gaiadr4.gaia_source`. Seuls `tap.schema` et `tap.row_limit` peuvent être ajustés ; c'est un écart technique, à documenter.
2. **Téléchargement**, en trois terminaux parallèles :
   - bash : `M=3 R=0 python run_wbt.py dr4 fetch` (puis R=1, R=2) ;
   - PowerShell : `$env:M=3; $env:R=0; python run_wbt.py dr4 fetch`.

   Le découpage est dimensionné sur le `COUNT(*)` serveur. Les tranches tronquées sont recoupées automatiquement. La complétude est vérifiée sur les identifiants uniques.
3. **Construction.** `python run_wbt.py dr4 build` : paires, cinématique, crible DR4, puis `all_source_flags` pour les seuls candidats.
4. **Analyse aveugle.** `python run_wbt.py dr4 blind` : contrôles sur N et sur la répartition en r/rM, S permuté ≈ 0, puis 5000 simulations Newton et 5000 MOND.
5. **Gel.** `python run_wbt.py dr4 freeze`, puis publier `FREEZE_dr4.json` par un commit horodaté **avant** la levée.
6. **Levée.** `python run_wbt.py dr4 unblind` produit `RESULT_dr4.json`, qui porte l'empreinte du gel. La levée est refusée si le code, la configuration, l'échantillon ou les simulations ont changé depuis le gel.
7. **Analyses secondaires.** `python run_wbt.py dr4 multiverse`, autorisé seulement si `RESULT` correspond au gel en cours.

Les fins de ligne sont imposées en LF (`.gitattributes`) pour que les empreintes SHA-256 soient identiques sur toutes les machines.

## Spécification principale (config `primary`)

- **Échantillon** : distance commune inférieure à 150 pc ; séparation 1–30 kau ; S/N parallaxe supérieur à 40.
- **Dégroupage** : 100 kau sur un catalogue à S/N supérieur à 10.
- **Qualité** : RUWE inférieur à 1,25 ; `ipd_frac_multi_peak` = 0 ; filtre HR (pente 3,3, hauteur 0,8 mag).
- **Vitesses radiales** : deux RV, |ΔRV| inférieur à 10 km/s.
- **Cinématique** : correction sphérique ; ṽ inférieur à 2,5 ; σ(ṽ) inférieur à 0,1.
- **Crible DR4 des compagnons cachés** : solution astrométrique et RV simples ; `unreliable_astrometry` faux ; `blended` = 0 ; ni SB2 ni raies dédoublées ; `rv_renormalised_gof` inférieur à 4 ; `non_single_star` = 0 ; `sea_env_flag` = 0.
- **Statistique** : S = médiane(ṽ | r/rM > 1) / médiane(ṽ | r/rM < 0,3) − 1.
- **Décision** : rejet d'une hypothèse si p < 0,00135 (3σ unilatéral), à la fois en simulation et en approximation gaussienne. Sinon le résultat est « non concluant ». En dessous de 1000 paires, l'étude est déclarée sous-dimensionnée.

## Répétition générale sur Gaia DR3 (16/10/2026)

| | Bac à sable (Linux) | Machine de l'auteur (Windows) |
|---|---|---|
| Catalogue | 1 426 271 étoiles, parallaxe > 6 mas | 2 877 863 étoiles, parallaxe > 4,5 mas |
| Paires (spéc. principale) | 1194 | 1189 |
| Séparation Newton/MOND attendue | 2,2σ | 2,1σ |
| S observé | +0,054 | +0,066 |
| p(Newton) / p(MOND) | 0,22 / 0,09 | 0,17 / 0,13 |
| Verdict | non concluant | non concluant |
| Multivers, S médian (108 spéc.) | +0,068 | +0,063 |

`RESULT_dr3.json`, `FREEZE_dr3.json`, `log_dr3.txt` et `multiverse_dr3.csv` proviennent de l'exécution sur la machine de l'auteur.

Lecture : avec la spécification pré-enregistrée, DR3 ne tranche pas. La valeur observée tombe à mi-chemin entre Newton (+0,005) et MOND (+0,147). Le χ² binné utilisé dans la littérature aurait pourtant affiché « Newton 6× ».

Dans le multivers, sans dégroupage, S ≈ +0,15, l'amplitude MOND exacte, produite par de la contamination. Avec dégroupage, S ≈ +0,05, un résidu inexpliqué que le crible direct de DR4 doit trancher.

L'exploration antérieure (multivers de 144 spécifications, test d'injection-récupération) se trouve dans `exploration_dr3/`. Elle est déclarée dans le pré-enregistrement.

## Changements v0.3

- **Encodage** : UTF-8 explicite partout ; le journal plantait sous Windows (cp1252).
- **Verrou du multivers** : désormais lié au manifeste de gel. En v0.2, il pouvait être contourné par un fichier `RESULT` étranger.
- **Catalogue** : garde-fou anti-doublons ; découpage dimensionné sur le comptage serveur ; complétude vérifiée sur les identifiants uniques.

## Non testé (DR4 indisponible)

- Nom de schéma `gaiadr4` et limites TAP de DR4.
- Format CSV des nouvelles colonnes.
- Requête `all_source_flags` par listes de `source_id`.
- Test d'accélération sur `epoch_astrometry` : à implémenter.

## Références

- Cookson, Banik, El-Badry et al. 2026, MNRAS, arXiv:2602.24035.
- Banik et al. 2024, MNRAS 527, 4573.
- Chae 2024, ApJ 960, 114.
- El-Badry & Rix 2018 ; El-Badry, Rix & Heintz 2021.

## Licence

MIT, voir `LICENSE`.
