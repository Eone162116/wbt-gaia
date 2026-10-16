# Validation du crible par époques (16/10/2026)

Données : astrométrie par époque Gaia DR4 de 12 sources, pré-publiée par l'ESA en juin 2026 (non redistribuée ici).
Téléchargement : https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip
Décompresser `GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml` dans ce dossier, puis lancer `python validate_epoch.py` (environ 3 min, nécessite astropy).

Résultats :
- `prerelease_classification.csv` : 12/12 sources bien classées (9 simples ou QSO passent, 3 binaires orbitales sont signalées).
- `injection_companions.csv`, `fig_epoch_detection.png` : probabilité de détection d'un compagnon caché selon son demi-grand axe a_in et son rapport de masse q (primaire 0,6 M☉ à 130 pc, échantillonnage réel d'une source G = 14,3). Le taux de faux positifs est de 0,25 %.

Crédit : ce travail utilise des données de la mission Gaia de l'ESA (https://www.cosmos.esa.int/gaia), traitées par le Gaia DPAC.
