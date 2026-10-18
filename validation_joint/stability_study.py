"""v0.9.1 — Critères de stabilité du test conjoint (amendement du 18/10/2026, fixé avant tout calcul de Λ v0.9.1 sur des
vitesses observées). Simulation seulement : géométrie, bruit et coupes de l'échantillon principal, aucune vitesse observée.
Les critères portent sur des jeux simulés et non sur Λ observé, pour pouvoir être évalués sur DR4 AVANT la levée, et parce
que DR3 et DR4 observent les mêmes paires (Λ observé sur DR3 ne doit pas entrer dans une décision sur DR4).
(c) dispersion due aux gabarits : pour chaque jeu simulé, écart-type de Λ entre 5 tirages indépendants de gabarits ;
    moyenne quadratique sur 20 jeux (10 Newton central, 10 MOND central) < 1,0 ;
(d) influence d'une paire : pour chaque jeu, max |ΔΛ| en excluant les paires une à une ; 90e centile sur les 20 jeux < 1,5.
À lancer depuis la racine du dépôt après « blind » : python validation_joint/stability_study.py [dr3|dr4]"""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from wbt_pipeline import select, hr_band, MG_SUN
import jointlik as jl

t0 = time.time(); REL = sys.argv[1] if len(sys.argv) > 1 else "dr3"
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]; CC = dict(CFG["calibration"]["contamination"])
CC["screen"] = CC.get("screen_by_release", {}).get(REL, CC["screen"]); GRID = "dr4" if REL == "dr4" else "dr3"
R = CFG["calibration"]["joint"]["R_templates"]
K = pd.read_pickle(f"K_{REL}.pkl"); hb, hs, _ = hr_band(K)
if P.get("hr_intercept") is not None: hb = P["hr_intercept"]
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (hb, hs, P["hr_height"]), **kw); S = K[m].reset_index(drop=True)
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
dmod = 5*np.log10(S.d.values/10)
G = dict(x=S.x.values, vc=S.vc.values, sv=np.stack([S.cv_aa, S.cv_ad, S.cv_dd, S.fd], 1), d=S.d.values, rsky=S.rsky.values, M=S.M.values,
         m1=mass(S.MG1.values), m2=mass(S.MG2.values), G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
del K, S                                                  # plus rien ne touche aux vitesses observées
E = pd.read_csv(f"eccfit_{REL}.csv"); F = pd.read_csv(f"finfer_{REL}.csv")
row = lambda h, z: E[(E.gabarits == h) & (E.zone == z)].iloc[0]
a_c = (row("newton", "bas").alpha, row("newton", "haut").alpha); b_c = (row("beta_separations", "bas").alpha, row("beta_separations", "haut").alpha)
f = {r.shape: (r.f_bas, r.f_central, r.f_haut) for r in F.itertuples()}
CONT = [("raghavan2010", 0.0), ("raghavan2010", f["raghavan2010"][1]), ("raghavan2010", f["raghavan2010"][2]), ("mdwarf_pokemon", f["mdwarf_pokemon"][1])]
zs = {"bas": np.where(G["x"] < 0.3)[0], "haut": np.where(G["x"] > 1)[0]}
rng = np.random.default_rng(910)                          # jeux d'ÉVALUATION (graine distincte des jeux de réglage)
mocks = [(g, jl.simulate_obs(G, a_c, b_c, g, f["raghavan2010"][1], "raghavan2010", CC, rng, P["vtmax"], P["dvsky"], grid=GRID))
         for g in [0]*10 + [1]*10]
L = np.zeros((5, len(mocks))); infl = np.zeros(len(mocks))
for s in range(5):
    T = jl.templates(G, zs, b_c, CONT, CC, np.random.default_rng(9100 + s), R=R, vtmax=P["vtmax"], dvsky=P["dvsky"], grid=GRID)
    for i, (g, (lv, ps, k)) in enumerate(mocks):
        parts = jl.lam_parts(lv, ps, k, zs, T); L[s, i] = jl.lam(lv, ps, k, zs, T, parts)
        if s == 0: infl[i] = np.abs(jl.loo_influence(lv, ps, k, zs, T, parts)[1]).max()
    print(f"gabarits {s + 1}/5 ({time.time()-t0:.0f} s)", flush=True)
sd = L.std(0, ddof=1); crit_c = float(np.sqrt((sd**2).mean())); crit_d = float(np.quantile(infl, 0.9))
out = pd.DataFrame(dict(hyp=["newton" if g == 0 else "mond" for g, _ in mocks], lam_moyen=L.mean(0), ecart_type_gabarits=sd, influence_max=infl))
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stability_{REL}.csv"), index=False)
print(out.round(2).to_string(index=False))
print(f"CRITÈRE (c) dispersion due aux gabarits < 1,0 : {crit_c:.2f} → {'OK' if crit_c < 1.0 else 'NON'} ; "
      f"(d) influence d'une paire (90e centile) < 1,5 : {crit_d:.2f} → {'OK' if crit_d < 1.5 else 'NON'}")
print("STABILITÉ :", "OK" if (crit_c < 1.0 and crit_d < 1.5) else "NON (le test conjoint reste en analyse secondaire)")
