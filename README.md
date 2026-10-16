# WBT pré-enregistré — version prête pour Gaia DR4 (v0.2, 16/10/2026)

Un seul point d'entrée, `run_wbt.py`, et un seul fichier de choix, `config_prereg.json`, qui est figé au dépôt du pré-enregistrement.

## Procédure du 2 décembre 2026

0. **Avant la sortie.** Déposer le pré-enregistrement en citant le hash du commit. Définir `WBT_BLIND_KEY` vers un fichier hors du dépôt : la clé n'est jamais affichée ni versionnée.
1. **Essai à sec.** Vérifier le nom du schéma (`SELECT TOP 1 source_id FROM gaiadr4.gaia_source`) et la limite de lignes synchrones. Seuls `tap.schema` et `tap.row_limit` peuvent être ajustés. C'est un écart technique, à documenter.
2. **Téléchargement.** `M=3 R=0 python run_wbt.py dr4 fetch &` (et R=1, R=2). Environ 3,4 M lignes (parallaxe > 4,5 mas, S/N > 10). Les tranches qui atteignent la limite sont recoupées automatiquement. Le total est vérifié contre un `COUNT(*)` serveur.
3. **Construction.** `python run_wbt.py dr4 build` : paires, cinématique, crible étoile DR4, puis `all_source_flags` (`non_single_star`, `sea_env_flag`) pour les seuls candidats.
4. **Analyse aveugle.** `python run_wbt.py dr4 blind`. On contrôle N, la répartition en r/rM et S permuté ≈ 0, puis on lance la calibration (5000 simulations Newton et 5000 MOND).
5. **Gel.** `python run_wbt.py dr4 freeze`, puis publier `FREEZE_dr4.json` (commit horodaté) **avant** la levée.
6. **Levée.** `python run_wbt.py dr4 unblind` donne `RESULT_dr4.json`. La levée est refusée si le code, la configuration, l'échantillon ou les simulations ont changé depuis le gel.
7. **Analyses secondaires.** `python run_wbt.py dr4 multiverse`, bloqué tant que l'aveuglement n'est pas levé.

## Spécification principale (config `primary`)

- **Échantillon** : distance commune inférieure à 150 pc ; séparation 1–30 kau ; S/N parallaxe supérieur à 40.
- **Dégroupage** : 100 kau sur un catalogue à S/N supérieur à 10.
- **Qualité** : RUWE inférieur à 1,25 ; `ipd_frac_multi_peak` = 0 ; filtre HR (pente 3,3, hauteur 0,8 mag).
- **Vitesses radiales** : deux RV, |ΔRV| inférieur à 10 km/s.
- **Cinématique** : correction sphérique ; ṽ inférieur à 2,5 ; σ(ṽ) inférieur à 0,1.
- **Crible DR4 des compagnons cachés** : `astrometry_origin` à solution simple ; `radial_velocity_origin` = `all_source_rvs` ; `unreliable_astrometry` faux ; `blended` = 0 ; ni SB2 ni raies dédoublées ; `rv_renormalised_gof` inférieur à 4 ; `non_single_star` = 0 ; `sea_env_flag` = 0.
- **Statistique** : S = médiane(ṽ | r/rM > 1) / médiane(ṽ | r/rM < 0,3) − 1.
- **Décision** : rejet d'une hypothèse si p < 0,00135 (3σ unilatéral), à la fois en simulation et en approximation gaussienne. Sinon le résultat est « non concluant ». En dessous de 1000 paires, l'étude est déclarée sous-dimensionnée.

## Répétition générale sur DR3 (16/10/2026)

Avec la spécification principale (le crible DR4 est neutre sur DR3), on obtient N = 1194 paires. La séparation attendue entre Newton et MOND n'est que de 2,2σ (σ(S) = 0,063).

Résultat après levée : S = +0,054, avec p(Newton) = 0,22 et p(MOND) = 0,09. Verdict : **non concluant**. Le χ² binné donnerait pourtant « Newton 11× » : c'est exactement le type de conclusion que la règle pré-enregistrée empêche.

Multivers (108 spécifications, statistique S) :

- sans dégroupage, S = +0,15 ± 0,04, soit précisément l'amplitude MOND attendue ;
- avec dégroupage, S = +0,05 ± 0,03 (de −0,015 à +0,09).

Le dégroupage retire une contamination qui imite quantitativement la marche MOND. Le résidu de +0,05, à mi-chemin, reste inexpliqué : contamination restante ou signal. DR3 ne permet pas de trancher, DR4 doit le permettre.

Garde-fous testés : une modification de `config_prereg.json` après le gel bloque la levée, et le multivers est refusé avant la levée.

## Non testé (DR4 indisponible)

- Nom de schéma `gaiadr4` et limites TAP de DR4.
- Format CSV des nouvelles colonnes (booléens, chaînes).
- Requête `all_source_flags` par listes de `source_id`.
- Test d'accélération sur `epoch_astrometry`, à implémenter (DataLink, sources finales seulement).
- Modèle d'erreurs DR4 : la calibration utilise les incertitudes réelles de l'échantillon, donc elle s'adapte automatiquement.
