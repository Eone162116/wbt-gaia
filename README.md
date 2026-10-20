# wbt-gaia — test des binaires larges (Newton vs MOND), pré-enregistré et aveuglé pour Gaia DR4

Version courante : v0.9.10 (20/10/2026). Auteur : Enguerrand Michon, chercheur indépendant (Lyon).

> **Spécification active** : `config_prereg.json` et le document de pré-enregistrement. Les entrées datées plus bas (v0.3 à v0.9.x) forment un historique : une description ancienne n'y vaut pas spécification.

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
2. **Téléchargement**, en trois terminaux parallèles, puis une passe finale (v0.9.7) :
   - PowerShell : `$env:M=3; $env:R=0; python fetch_gaia.py dr4` (puis R=1, R=2) ; bash : `M=3 R=0 python fetch_gaia.py dr4` ;
   - une fois les trois terminés : `python run_wbt.py dr4 fetch`, sans M ni R. Cette passe complète ce qui manque, vérifie la complétude et consigne le serveur interrogé dans `ACQUISITION_dr4.json`. Seule son erreur est bloquante.

   Le découpage est dimensionné sur le `COUNT(*)` serveur. Les tranches tronquées sont recoupées automatiquement. La complétude est vérifiée sur les identifiants uniques.
3. **Construction.** `python run_wbt.py dr4 build` : paires, cinématique, crible DR4, puis `all_source_flags` pour les seuls candidats.
3b. **Crible par époques.** `python run_wbt.py dr4 epoch` : astrométrie par époque (DataLink) des composantes candidates, tests d'accélération et UWE (`epoch_accel.py`). Si DataLink échoue, télécharger depuis l'archive et indiquer le fichier via `WBT_EPOCH_FILE`.
4. **Analyse aveugle.** `python run_wbt.py dr4 blind` : contrôles sur N et sur la répartition en r/rM, S permuté ≈ 0, puis 50 000 simulations par colonne (Newton, MOND et scénarios d'excentricité), l'enveloppe de contamination et la calibration du test conjoint.
5. **Gel.** `python run_wbt.py dr4 freeze`, puis publier `FREEZE_dr4.json` et `ACQUISITION_dr4.json` par un commit horodaté **avant** la levée, et vérifier que le push a abouti : après `git fetch`, `git rev-parse HEAD` et `git rev-parse origin/main` doivent coïncider.
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
- **Hypothèse nulle newtonienne (v0.5)** : Newton pur **et** Newton contaminé par des compagnons cachés, pour une fraction de composantes de 0 à 50 % (crible époques + image pessimiste, `contamination.py`). « Newton rejeté » exige le rejet pour chaque valeur de l'enveloppe. Les simulations MOND restent pures, ce qui est prudent puisque la contamination pousse S vers le haut.

## Répétition générale sur Gaia DR3 (16/10/2026)

> Résultats historiques, obtenus avec la version de l'époque (v0.3). Ils ne se transposent pas aux versions postérieures, et en particulier pas à partir de la v0.9.9, dont les distributions nulles diffèrent. Les résultats de référence d'une version sont dans `RESULT_dr3.json` et `log_dr3.txt` au commit correspondant.

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

## Crible par astrométrie par époque (v0.4)

Chaque composante est ajustée par un modèle simple (5 paramètres), avec accélération (7) puis avec variation de l'accélération (9), sur les positions le long du balayage moyennées par transit. Une composante est signalée si un test F d'accélération donne p < 0,001 ou si l'UWE du modèle simple dépasse 1,4. Une binaire large de quelques kau produit environ 1e-4 mas/an² à 130 pc, indétectable : toute détection signale un compagnon proche.

Validation sur la pré-publication DR4 (12 sources, juin 2026), dans `validation_epoch/` :

- les 9 sources simples ou QSO passent, et les parallaxes sont retrouvées à mieux que 1 % ;
- les 3 binaires orbitales sont signalées : HD 114762 et Gaia-4 par l'UWE (3,0 et 2,5), Gaia BH3 par l'accélération (12σ) ;
- le taux de faux positifs est de 0,25 % ;
- injection de compagnons cachés (primaire 0,6 M☉ à 130 pc) : détection d'au moins 50 % pour a_in ≈ 0,1–20 au (0,2–15 au si q = 0,1).

Angle mort restant : au-delà d'environ 30 au, l'accélération devient trop faible sur 5,5 ans. À 130 pc, le crible doit alors être pris en charge par les drapeaux image (`ipd_frac_multi_peak`, `sea_env_flag`, `nss_resolved_pair`), dont la limite de résolution reste à quantifier.

## Fraction de compagnons cachés mesurée sur les données (v0.6)

L'enveloppe newtonienne ne balaie plus f_in de 0 à 0,5 : f_in est mesuré à partir de la fraction de paires candidates signalées. Sur DR3, c'est la fraction avec RUWE > 1,25 ; sur DR4, la fraction signalée par le crible par époques. Ces paires candidates passent toutes les coupes indépendantes des vitesses : la mesure respecte donc l'aveuglement. L'inversion utilise le modèle de détection (`finfer.py`, `validation_contamination/pgrid*.npy`) pour deux formes de population : type solaire (Raghavan et al. 2010) et naines M (POKEMON, pic à 7,9 au). La bande est ± 2σ binomiale, élargie sur le taux de faux positifs r0.

Répétition DR3 : 22,9 % des 2021 paires candidates signalées ; f_in = 0,23–0,38 (type solaire) ou 0,14–0,23 (naines M). Le biais newtonien maximal passe de +0,055 (f_in = 0,5) à environ +0,03.

Effet attendu sur DR4 : à N = 5000, la probabilité de rejeter Newton si MOND est vrai passe de 0,09 à 0,37 ; à N = 10 000, de 0,41 à 0,86. La probabilité de rejeter MOND ne dépend pas de l'enveloppe.

Essais écartés : une marche fondée sur le quartile inférieur de ṽ au lieu de la médiane donne le même biais avec plus de bruit. La médiane est conservée.

## Changements v0.7 (audit externe du 17/10/2026)

- **v0.7-A** : équation MOND conforme à Cookson et al. 2026 (éq. 7). Le terme de champ externe est [tanh(0,825 g_N,e/g_N,i)]^3,7, et non tanh((·)^3,7). L'écart sur le boost allait jusqu'à 1,5 % vers x ≈ 1. Test de conformité : `python tests/test_mond_eq7.py`. Les scripts archivés de `exploration_dr3/` gardent l'ancienne forme, comme trace de l'exploration.
- **v0.9.10** : corrections de l'audit externe du 20/10/2026 sur la v0.9.9. Le tirage conditionnel va jusqu'à 200 essais par paire, et chaque calibration (enveloppe de S, test conjoint) compte les tirages de paires non reconstitués, consignés dans le journal. Règle fixée avant les données : au-delà de 0,1 %, la calibration est refusée (`calibration.conditionnel` dans la configuration). La documentation précise les coupes effectivement simulées, signale que les distributions nulles de S et de Λ changent depuis la v0.9.9, que les gabarits du test conjoint restent des mélanges pondérés par la survie (approximation, Λ secondaire), et que les résultats antérieurs sont historiques. Test : `tests/test_conditional.py` (6 cas, dont le refus d'une calibration incomplète).
- **v0.9.9** : jeux simulés conditionnés à la sélection. La géométrie observée a déjà passé le crible de contamination et les coupes ; les simulations ajoutaient des compagnons puis retiraient ceux que le crible détectait, si bien qu'elles gardaient 65 à 76 % des paires observées (DR3). Désormais, chaque paire est retirée jusqu'à passer les coupes simulées, et elles seules : crible de contamination simulé, ṽ < vtmax, Δv sur le ciel (`jointlik.simulate_obs_cond`, `contamination.contaminated_newton_cond`) : N simulé = N observé. Concerne l'enveloppe de contamination de S et la calibration du test conjoint, dont les distributions nulles changent (largeur, et possiblement position). Les simulations pures de S et les gabarits sont inchangés ; aucun seuil ne change. Test : `tests/test_conditional.py`. Exploration : `exploration_dr3/decomposition_lambda.py`.
- **v0.9.8** : corrections de l'audit externe du 20/10/2026 sur la v0.9.7. `ACQUISITION_dr4.json` devient obligatoire en DR4, avec ses sections catalogue, drapeaux et époques exigées avant `blind` et `freeze`. Le gel et la levée comparent les ensembles exacts de fichiers, si bien qu'un fichier gelé ne peut plus disparaître sans que la levée soit refusée. Le catalogue est inventorié (fichier, taille, SHA-256 de chaque tranche, requête, serveur), et le journal garde l'historique des réécritures. Règle fixée avant les données : plus de 0,5 % de drapeaux absents après deux requêtes TAP entraîne un arrêt. L'exercice refuse les fichiers non suivis et vérifie que les fichiers requis sont suivis par Git. Tests : `tests/test_acquisition.py` (9 cas) et `tests/test_freeze.py` (4 cas). Rappel : depuis la v0.9.7, un drapeau inconnu compte comme non propre, ce qui peut changer la population retenue en DR4 sans qu'aucun seuil ne change.
- **v0.9.7** : corrections de l'audit externe du 20/10/2026 sur la v0.9.6 et la fiche jour J. Drapeaux de repli lus strictement (colonnes, doublons, identifiants manquants : arrêt) et drapeau inconnu compté non propre ; `ACQUISITION_<rel>.json` consigne le serveur TAP, les fichiers de repli et leurs empreintes, et il est gelé ; téléchargement parallèle par `fetch_gaia.py`, avec une seule vérification finale ; exercice DR4 dans un dossier horodaté jamais effacé, sur un dépôt propre, avec copie du rapport dans `validation_dr4_exercice/` ; tests fonctionnels `tests/test_acquisition.py`. Aucun changement de méthode : S reste le test principal.
- **v0.9.6** : préparation du jour J. `tools/exercice_dr4.py` fait tourner toute la branche DR4 (build avec drapeaux, epoch, blind en mode complet, freeze, unblind, multiverse) sur un faux DR4 fabriqué à partir du catalogue DR3, dans un dossier isolé hors du dépôt ; l'option `--fumee` en fait un test de plomberie d'environ 20 minutes. Plans B : `WBT_TAP_URL` pour interroger un miroir officiel si l'archive ESA est saturée (complétude vérifiée contre ce même serveur), `WBT_FLAGS_FILE` pour les drapeaux de all_source_flags et `WBT_EPOCH_FILE` pour l'astrométrie par époque, tous deux téléchargés à part.
- **v0.9.5** : études de caractérisation du test conjoint refaites sur les entrées de la répétition DR3 complète (α = 2,00 ; 0,50) : (c) 0,97, (d) 1,56, gain +0,51, biais 0,52σ ; décision inchangée, sorties du mode rapide archivées. Multivers DR3 (108 spécifications) : S médian +0,073, spécification principale au 58e centile ; le dégroupage domine (sans lui, S ≈ +0,15).
- **v0.9.4** : garde de complétude. « blind » écrit `BLIND_<rel>_complete.json` (empreintes des fichiers à geler) seulement s'il va au bout, et « freeze » refuse de geler si ce marqueur manque ou si un fichier a changé depuis. Constat à l'origine : le 19/10/2026, un « freeze » lancé après un « blind » interrompu a gelé des artefacts incohérents (commit 3e30878, répétition DR3 sans effet sur le protocole DR4).
- **v0.9.3** : corrections issues de l'audit externe du 19/10/2026. `jointlik.py` profilait η au pas de 0,01 dex au lieu des 0,005 gelés, car un ancien bloc de constantes écrasait `ETA`. C'est corrigé, et les trois études relancées à l'identique donnent les mêmes conclusions : (c) 0,97, (d) 1,72, gain +0,64, biais 0,50σ. Les anciennes sorties sont archivées. Le chargement passe par `validation_joint/common.py` au lieu d'un `exec` ; chaque sortie a son fichier de provenance (commit, graines, SHA-256) ; le patch refuse d'écraser un contenu inconnu et ses contrôles finaux sont bloquants. `joint_power.py` est étiqueté comme étude historique. S reste le test principal ; la promotion n'est pas rouverte.
- **v0.9.2** : question du test conjoint close. S est le test principal de DR3 et de DR4, sans réexamen sur la géométrie DR4. Diagnostic sur simulations (`validation_joint/power_origin.py`) : la puissance du test conjoint vient surtout des paires de la zone haute dont ṽ est compris entre 1 et √2. C'est aussi là que les compagnons cachés déposent des paires, si bien que sa puissance et sa sensibilité à la contamination ont la même source. Ni un plancher plus haut ni une troncature à √2 n'y changent rien.
- **v0.9.1** : test conjoint lissé et continu (`jointlik.py`). Les gabarits deviennent des densités fines lissées, la vraisemblance est interpolée au point observé, η et α sont profilés finement, et toutes les configurations d'une zone partagent les mêmes tirages aléatoires. Λ devient une fonction continue des données. Les hyperparamètres sont fixés sur simulations avant tout calcul sur données réelles. Les critères de stabilité sont évalués sur simulations (`validation_joint/stability_study.py`). Correctif de téléchargement : chaque tranche est comparée au compte du serveur, `python fetch_gaia.py repair <rel>` supprime les tranches incomplètes, et `fetch` s'arrête si le catalogue est incomplet. Résultat de stabilité sur DR3 : dispersion due aux gabarits 0,97 (seuil 1,0), mais influence d'une paire 1,72 au 90e centile (seuil 1,5) : la v0.9.1 n'est pas promue, S reste le test principal, et le test conjoint reste une analyse secondaire pré-enregistrée.
- **v0.9a** : promotion du test conjoint **suspendue**. Sur deux répétitions DR3 qui ne diffèrent que de quelques paires, Λ observé vaut −15,4 et +5,2 ; une seule paire peut le déplacer de 6,8, à cause des cases et de la grille discrète de η. S redevient le test principal, et le test conjoint reste rapporté. La v0.9.1 (vraisemblance lissée et continue) ne sera promue qu'avec des critères de stabilité en plus du gain de puissance et de la robustesse.
- **v0.9** : le test conjoint devient le test principal. Ce test calcule un rapport de vraisemblance profilé sur la norme et la direction des vitesses (`jointlik.py`). Il profile l'excentricité par zone, la contamination et une échelle globale η, et ignore donc la normalisation absolue des masses. Sa promotion suit une règle fixée avant tout calcul sur données réelles : gain de puissance d'au moins 0,10 et biais de robustesse inférieur à 0,5σ. Sur DR3, le gain vaut +0,12 et le biais maximal 0,38σ (`validation_joint/promotion_study.py`). Le test S reste rapporté comme analyse secondaire (`verdict_S`).
- **v0.8.1** : la pente β de la distribution des demi-grands axes (dN/da ∝ a^-β) est mesurée sans vitesses sur les séparations projetées de chaque zone de x. Elle entre dans les trois scénarios mesurés : gradient minimal pour Newton, maximal pour MOND. β ne change pas ψ, mais un β qui s'accentue avec la séparation abaisse S (environ −0,035 pour 1,0→1,6). Sur DR3 : β ≈ 1,5 vers 1–2,5 kau et ≈ 1,9–2,2 vers 7–30 kau.
- **v0.8** : excentricité mesurée.
    - **Mesure** : à l'étape aveugle, l'indice d'excentricité α est mesuré dans chaque zone de x par les angles entre séparation et vitesse relative (`eccfit.py`). Ces angles ne dépendent pas de |ṽ| dans le modèle opérationnel.
    - **Scénarios** : trois scénarios issus de la mesure (central, le plus favorable à Newton, le plus favorable à MOND) entrent dans les deux hypothèses nulles. L'enveloppe de contamination utilise le scénario le plus favorable à Newton.
    - **Motivation** : sur DR3, α ≈ 1,5 à x < 0,3 et α ≈ 0,5 à x > 1. Sous Newton, ce gradient imite une marche MOND d'environ +0,085.
- **v0.7-D** : verrouillage et contrôles.
    - **Gel** : il couvre aussi les données analysées (`K_<rel>.pkl`, `sample_<rel>.csv`), les contrôles avant levée et, sur DR4, les statistiques du crible par époques.
    - **Arrêt bloquant** : sur DR4, l'analyse s'arrête si le crible par époques exigé est absent.
    - **Contrôles de la section 14** : ils sont exécutés et archivés. Avant la levée, dans `CONTROLS_<rel>_aveugle.json`, qui est gelé : permutation, injection-récupération, effectifs par zone, comparaison des deux mesures de f_in. À la levée, dans `RESULT_<rel>.json` : contrôle en régime newtonien, c'est-à-dire de forte accélération, à x < 0,3.
- **v0.7-B** : recalibration.
    - **Volume** : 50 000 simulations par hypothèse pure et 20 000 par scénario de contamination, vectorisées par lots.
    - **Règle de rejet** : la borne haute de Clopper-Pearson à 95 % de la probabilité de queue et l'approximation gaussienne doivent toutes deux passer sous 0,00135. Une seule simulation extrême ne peut plus renverser un verdict.
    - **Excentricités** : deux scénarios où l'excentricité augmente avec la séparation, α de 1,0 à 1,3 et de 1,0 à 1,6 entre 1 et 30 kau (Hwang, Ting & Zakamska 2022). Chaque hypothèse est jugée dans le scénario qui lui est le plus favorable : Newton thermique et contaminé (S le plus haut), MOND superthermique (S le plus bas).
    - **Effectifs** : au moins 100 paires exigées dans chaque zone de x.
    - **Seuil σ(ṽ)** : choisi par simulation seulement (`validation_calibration/sigma_threshold_study.py`). Sur DR3, le relâcher de 0,1 à 0,3 n'ajoute que 9 paires, et 0,1 est conservé.
    - **Durée** : l'étape `blind` prend environ 50 minutes. `WBT_QUICK=1` divise les simulations par 10 pour les répétitions, et ce mode est interdit sur DR4.
- **v0.7-C** : sélection. σ(ṽ) vient désormais de la covariance complète de Δμ, avec `pmra_pmdec_corr` et une moyenne en direction, sans le terme proportionnel à la vitesse observée : la coupe σ(ṽ) < 0,1 ne dépend plus des vitesses. Les simulations tirent leur bruit dans cette covariance, avec en plus l'erreur d'échelle due à la distance. Les candidats du crible par époques sont choisis sans aucune coupe en vitesse. Chaque paire a l'un de trois états (propre, signalé, manquant), et f_in est mesuré sur les seules paires évaluées.

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
