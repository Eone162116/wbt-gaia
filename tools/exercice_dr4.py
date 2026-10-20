"""Exercice DR4 à blanc (v0.9.6).
Fait tourner toute la branche DR4 du pipeline sur un FAUX DR4 fabriqué à partir du catalogue DR3 local :
  build (flags lus dans un fichier), epoch (astrométrie par époque synthétique), blind en mode complet, freeze, unblind, multiverse.
Tout se passe dans un dossier isolé, hors du dépôt, avec une clé d'aveuglement propre à l'exercice. Rien n'est envoyé au réseau ;
rien n'est écrit dans le dépôt. Les colonnes DR4, les drapeaux et l'astrométrie par époque sont inventés : les résultats n'ont
AUCUNE valeur physique. Seule compte la question « la chaîne DR4 va-t-elle au bout, et en combien de temps ? ».

Usage, depuis la racine du dépôt, après un fetch DR3 complet :
  python tools/exercice_dr4.py            exercice complet, comptes de simulation du protocole (≈ 4 h, à lancer le soir)
  python tools/exercice_dr4.py --fumee    comptes de simulation divisés par 100 (≈ 20 min) : plomberie seulement
Dossier créé, jamais réutilisé ni effacé : ../wbt-gaia-EXERCICE-DR4-<date-heure>[-fumee] (v0.9.7, audit F6).
Le dépôt doit être propre (aucune modification des fichiers suivis) : la copie est identifiée par son commit (audit F7) ;
--autoriser-modifs lève ce refus et consigne alors l'état réel (git status et empreinte de chaque fichier copié).
Rapport et journal : dans le dossier d'exercice, et copiés dans validation_dr4_exercice/ du dépôt (seule écriture dans le
dépôt), pour servir de preuve d'exécution (audit F1)."""
import os, sys, json, glob, shutil, subprocess, time, datetime
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.getcwd())
if not os.path.exists(os.path.join(ROOT, "config_prereg.json")) or not os.path.exists(os.path.join(ROOT, "run_wbt.py")):
    sys.exit("Lance ce script depuis la racine du dépôt wbt-gaia.")
FUMEE = "--fumee" in sys.argv
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DST = os.path.abspath(os.path.join(ROOT, os.pardir, f"wbt-gaia-EXERCICE-DR4-{STAMP}" + ("-fumee" if FUMEE else "")))
if os.path.exists(DST): sys.exit(f"ARRÊT : {DST} existe déjà ; rien n'est effacé, relance dans une seconde.")
if os.path.commonpath([DST, ROOT]) == ROOT: sys.exit("ARRÊT : le dossier d'exercice ne doit pas être dans le dépôt.")
DR3 = sorted(glob.glob(os.path.join(ROOT, "dr3_*.csv")))
if not DR3: sys.exit("Catalogue DR3 local absent : lance d'abord « python run_wbt.py dr3 fetch ».")
rng = np.random.default_rng(20261202)
import hashlib
DIRTY = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
if DIRTY and "--autoriser-modifs" not in sys.argv:
    sys.exit("ARRÊT : le dépôt a des modifications non validées ; l'exercice doit tourner sur un commit exact.\n" + DIRTY
             + "\nValide-les (git commit) ou relance avec --autoriser-modifs (l'état réel sera consigné).")
T0 = time.time(); REP = dict(debut=datetime.datetime.now().isoformat(timespec="seconds"), mode="fumée" if FUMEE else "complet",
                             dossier=DST, etapes={}, controles={})
JOURNAL = os.path.join(DST, "EXERCICE_DR4_journal.txt")
def say(*a):
    msg = " ".join(str(x) for x in a); print(msg, flush=True)
    with open(JOURNAL, "a", encoding="utf-8") as fh: fh.write(msg + "\n")
def finish(ok, why=""):
    REP.update(fin=datetime.datetime.now().isoformat(timespec="seconds"), duree_totale_min=round((time.time() - T0)/60, 1),
               succes=ok, motif=why)
    json.dump(REP, open(os.path.join(DST, "EXERCICE_DR4_rapport.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    proof = os.path.join(ROOT, "validation_dr4_exercice"); os.makedirs(proof, exist_ok=True)     # v0.9.7 : preuve d'exécution (audit F1)
    tag = f"{STAMP}_{REP['mode'].replace('é', 'e')}"
    shutil.copy2(os.path.join(DST, "EXERCICE_DR4_rapport.json"), os.path.join(proof, f"rapport_{tag}.json"))
    if os.path.exists(JOURNAL): shutil.copy2(JOURNAL, os.path.join(proof, f"journal_{tag}.txt"))
    say(("EXERCICE RÉUSSI" if ok else "EXERCICE ÉCHOUÉ : " + why) + f" — {REP['duree_totale_min']} min ; rapport : EXERCICE_DR4_rapport.json")
    sys.exit(0 if ok else 1)

# 1. Dossier isolé : copie du code et des fichiers suivis par Git (sans les données)
os.makedirs(DST)
tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
for f in tracked:
    os.makedirs(os.path.join(DST, os.path.dirname(f)), exist_ok=True); shutil.copy2(os.path.join(ROOT, f), os.path.join(DST, f))
for f in glob.glob(os.path.join(DST, "*_dr3*")) + glob.glob(os.path.join(DST, "FREEZE_*")) + glob.glob(os.path.join(DST, "RESULT_*")):
    if os.path.isfile(f): os.remove(f)
REP["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
REP["depot_modifie"] = DIRTY or False
REP["empreintes_copiees"] = {f: hashlib.sha256(open(os.path.join(DST, f), "rb").read()).hexdigest() for f in tracked if os.path.exists(os.path.join(DST, f))}
REP["avertissement"] = ("Faux DR4 : vitesses, positions et parallaxes DR3 réelles, filtrées par des drapeaux et une astrométrie par époque "
                        "inventés. Le S de l'exercice n'a aucune valeur physique. Une VOTable synthétique lisible ne valide pas le format réel "
                        "de DataLink DR4, à contrôler le jour J (audit F10).")
say(f"EXERCICE DR4 À BLANC ({REP['mode']}) — dossier {DST} — commit {REP['commit'][:7]} — résultats sans valeur physique")
if FUMEE:
    c = json.load(open(os.path.join(DST, "config_prereg.json"), encoding="utf-8")); C = c["calibration"]
    C["n_mocks"] //= 100; C["contamination"]["n_mocks_per_f"] //= 100; C["ecc_fit"]["R"] = max(5, C["ecc_fit"]["R"]//20)
    C["joint"]["R_templates"] = max(6, C["joint"]["R_templates"]//15); C["joint"]["n_par_scenario"] //= 100
    json.dump(c, open(os.path.join(DST, "config_prereg.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    say("mode fumée : comptes de simulation réduits dans la COPIE de la configuration (le dépôt n'est pas touché)")

# 2. Faux catalogue DR4 : tranches DR3 + colonnes DR4 inventées ; fichier de drapeaux all_source_flags inventé
t = time.time(); flags = []
for f in DR3:
    df = pd.read_csv(f); n = len(df); hasrv = df.radial_velocity.notna().values
    df["astrometry_origin"] = np.where(rng.random(n) < 0.97, "all_source_astrometry", "nss_two_body_orbit")
    df["radial_velocity_origin"] = np.where(hasrv, np.where(rng.random(n) < 0.98, "all_source_rvs", "nss_rv"), None)
    df["unreliable_astrometry"] = rng.random(n) < 0.01
    df["blended"] = (rng.random(n) < 0.02).astype(int)
    df["rv_assumed_sb2"] = rng.random(n) < 0.005
    df["fraction_double_lined"] = np.where(rng.random(n) < 0.01, 0.3, 0.0)
    df["rv_renormalised_gof"] = np.where(hasrv, np.abs(rng.normal(1.0, 0.8, n)), np.nan)
    df.to_csv(os.path.join(DST, os.path.basename(f).replace("dr3_", "dr4_", 1)), index=False)
    flags.append(pd.DataFrame(dict(source_id=df.source_id, non_single_star=(rng.random(n) < 0.04).astype(int),
                                   sea_env_flag=(rng.random(n) < 0.02).astype(int))))
FLAGS = os.path.join(DST, "EXERCICE_flags_dr4.csv"); pd.concat(flags).to_csv(FLAGS, index=False); del flags
REP["etapes"]["faux_catalogue"] = dict(minutes=round((time.time() - t)/60, 1), tranches=len(DR3))
say(f"faux catalogue DR4 : {len(DR3)} tranches ({REP['etapes']['faux_catalogue']['minutes']} min)")

ENV = dict(os.environ, WBT_BLIND_KEY=os.path.join(DST, "EXERCICE_cle_aveuglement.txt"), WBT_FLAGS_FILE=FLAGS,
           WBT_EPOCH_FILE=os.path.join(DST, "EXERCICE_epoques_dr4.xml"), PYTHONIOENCODING="utf-8")
ENV.pop("WBT_QUICK", None); ENV.pop("WBT_TAP_URL", None)
def stage(name):
    t = time.time(); say(f"--- étape {name} ---")
    with open(os.path.join(DST, f"EXERCICE_sortie_{name}.txt"), "w", encoding="utf-8") as out:
        p = subprocess.run([sys.executable, "run_wbt.py", "dr4", name], cwd=DST, env=ENV, stdout=out, stderr=subprocess.STDOUT)
    tail = open(os.path.join(DST, f"EXERCICE_sortie_{name}.txt"), encoding="utf-8", errors="replace").read().strip().splitlines()[-3:]
    REP["etapes"][name] = dict(minutes=round((time.time() - t)/60, 1), code_retour=p.returncode, fin_sortie=tail)
    for l in tail: say("   ", l[:200])
    say(f"    {name} : code {p.returncode}, {REP['etapes'][name]['minutes']} min")
    if p.returncode != 0: finish(False, f"étape {name} en erreur (voir EXERCICE_sortie_{name}.txt)")

# 3. build, puis astrométrie par époque synthétique pour les composantes candidates
stage("build")
sys.path.insert(0, DST); os.chdir(DST)
from wbt_pipeline import select, hr_band
cfg = json.load(open("config_prereg.json", encoding="utf-8")); P = cfg["primary"]; Gm = cfg["multiverse"]
K = pd.read_pickle("K_dr4.pkl"); hb, hs, _ = hr_band(K)
cand, _ = select(K, (hb, hs, 0.8), hr=False, sph=True, dmax=max(Gm["dmax"]), ruwe=1e9, degroup=None, screen="dr3", ipd=False,
                 epoch=False, vtmax=1e9, dvsky=1e9, drv=1e9, svt=P["svt"])
ids = np.unique(np.concatenate([K.sid1[cand], K.sid2[cand]])); del K
t = time.time(); NT = 20 if FUMEE else 40; NCCD = 9; NS = 1e9*86400*365.25
keep_ids = ids[rng.random(len(ids)) > 0.05]                   # ≈ 5 % sans données d'époque : état « manquant »
acc = rng.random(len(keep_ids)) < 0.15                        # ≈ 15 % avec une accélération détectable : état « signalé »
from astropy.table import Table
ns, R = len(keep_ids), len(keep_ids)*NT                        # génération vectorisée : une ligne par transit, 9 CCD par ligne
src = np.repeat(np.arange(ns), NT)
tau = rng.uniform(-3.0, 2.6, R); th = rng.uniform(0, 2*np.pi, R); fw = rng.uniform(-1, 1, R)
pm = rng.normal(0, 50, (ns, 2))[src]; plx = rng.uniform(5, 30, ns)[src]
gacc = (rng.normal(0, 1.5, (ns, 2))*acc[:, None])[src]
sig = rng.uniform(0.15, 0.6, ns)[src]; gm = rng.uniform(8, 17, ns)[src]
w = ((pm[:, 0]*tau + 0.5*gacc[:, 0]*tau**2)*np.sin(th) + (pm[:, 1]*tau + 0.5*gacc[:, 1]*tau**2)*np.cos(th) + plx*fw)
tab = Table(dict(source_id=keep_ids.astype(np.int64)[src], used_by_agis_al=rng.random((R, NCCD)) > 0.03,
                 centroid_pos_al=w[:, None] + rng.normal(0, 1, (R, NCCD))*sig[:, None], centroid_pos_error_al=np.repeat(sig[:, None], NCCD, 1),
                 agis_source_excess_noise=np.zeros(R), obs_time_tcb=(tau + 7.5)[:, None]*NS + np.arange(NCCD)[None, :]*4.85e9,
                 obs_time_bary_corr=np.zeros(R), scan_pos_angle=np.repeat(np.degrees(th)[:, None], NCCD, 1),
                 parallax_factor_al=fw, g_mag=gm))
tab.write(ENV["WBT_EPOCH_FILE"], format="votable", overwrite=True); del tab
REP["etapes"]["epoques_synthetiques"] = dict(minutes=round((time.time() - t)/60, 1), composantes_candidates=int(len(ids)),
                                             avec_donnees=int(len(keep_ids)), acceleration_injectee=int(acc.sum()), transits_par_source=NT)
say(f"astrométrie par époque synthétique : {len(keep_ids)} composantes sur {len(ids)}, {acc.sum()} avec accélération injectée "
    f"({REP['etapes']['epoques_synthetiques']['minutes']} min)")

# 4. Toute la chaîne DR4
for s in ("epoch", "blind", "freeze", "unblind", "multiverse"): stage(s)

# 5. Contrôles de bout de chaîne
log = open("log_dr4.txt", encoding="utf-8", errors="replace").read() if os.path.exists("log_dr4.txt") else ""
R = json.load(open("RESULT_dr4.json", encoding="utf-8")) if os.path.exists("RESULT_dr4.json") else {}
ctrl = json.load(open("CONTROLS_dr4_aveugle.json", encoding="utf-8")) if os.path.exists("CONTROLS_dr4_aveugle.json") else {}
REP["controles"] = {
    "crible par époques appliqué": "crible par époques :" in log and "epoch_stats_dr4.csv" in os.listdir("."),
    "flags lus depuis le fichier": "flags lus dans" in log,
    "f_in mesuré avec le drapeau DR4": "drapeau dr4" in log,
    "comparaison f_in époques/RUWE faite": isinstance(ctrl.get("f_in_comparaison"), dict),
    "mode complet (pas de mode rapide)": "MODE RAPIDE" not in log,
    "marqueur de complétude écrit": os.path.exists("BLIND_dr4_complete.json"),
    "gel et levée": bool(R.get("freeze_manifest")) and "verdict" in R,
    "test conjoint calculé": "conjoint" in R,
    "multivers DR4 (crible et époques inclus)": os.path.exists("multiverse_dr4.csv"),
    "sources de données consignées et gelées": os.path.exists("ACQUISITION_dr4.json")
        and "ACQUISITION_dr4.json" in json.load(open("FREEZE_dr4.json", encoding="utf-8")).get("files", {})}
REP["resultat_sans_valeur_physique"] = dict(N=R.get("N"), verdict=R.get("verdict"), S_obs=R.get("S_obs"))
for k, v in REP["controles"].items(): say(f"  {'OK ' if v else 'NON'} {k}")
finish(all(REP["controles"].values()), "" if all(REP["controles"].values()) else "contrôle(s) de bout de chaîne non satisfait(s)")
