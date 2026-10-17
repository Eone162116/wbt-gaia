"""v0.8 — Validation de la mesure d'excentricité par les angles v–r (eccfit.py), par simulation seulement.
1. Injection-récupération sur la géométrie de l'échantillon principal : α connu (thermique, croissant, décroissant,
   avec ou sans boost MOND) → α estimé.
2. Indépendance de ψ vis-à-vis de la distribution des demi-grands axes (dN/da ∝ a^-β), qui modifie en revanche la
   médiane de ṽ (nuisance identifiée le 17/10/2026, à traiter en v0.8.1).
À lancer depuis la racine du dépôt après « build » : python validation_eccentricity/ecc_injection.py [dr3|dr4]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from wbt_pipeline import select, hr_band, observe
import eccfit as ef
from calibrate import mond_boost3d

REL = sys.argv[1] if len(sys.argv) > 1 else "dr3"
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]
K = pd.read_pickle(f"K_{REL}.pkl"); hb, hs, _ = hr_band(K)
if P.get("hr_intercept") is not None: hb = P["hr_intercept"]
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (hb, hs, P["hr_height"]), **kw); S = K[m]
x, vc = S.x.values, S.vc.values; sv = np.stack([S.cv_aa, S.cv_ad, S.cv_dd, S.fd], 1)
rng = np.random.default_rng(2026); out = []
zones = {"bas": np.where(x < 0.3)[0], "haut": np.where(x > 1)[0]}
TPL = {z: ef.templates(i, x, vc, sv, rng, "newton", 100, P["vtmax"], P["dvsky"]) for z, i in zones.items()}

def mock(idx, alpha, mond):
    rs, vt, ratio = ef._sky_rv(len(idx), alpha, rng)
    if mond: vt = vt*mond_boost3d(x[idx]*ratio)[:, None]
    v = observe(vt*vc[idx][:, None], sv[idx], rng)
    va = np.linalg.norm(v, axis=1); keep = (va/vc[idx] < P["vtmax"]) & (va < P["dvsky"])
    c = np.abs((rs*v).sum(1))/np.linalg.norm(rs, axis=1)/np.maximum(va, 1e-12)
    return np.degrees(np.arccos(np.clip(c, 0, 1)))[keep]

for lab, (al, ah), mond in (("thermique", (1.0, 1.0), False), ("croissant 1,0→1,3", (1.0, 1.3), False),
                            ("décroissant 1,5→0,5", (1.5, 0.5), False), ("décroissant + MOND", (1.5, 0.5), True)):
    for _ in range(int(os.environ.get("WBT_STUDY_R", "10"))):
        lo = ef.fit(mock(zones["bas"], al, mond), TPL["bas"]); hi = ef.fit(mock(zones["haut"], ah, mond), TPL["haut"])
        out.append(dict(cas=lab, vrai_bas=al, vrai_haut=ah, est_bas=lo[1], est_haut=hi[1],
                        couvre_bas=lo[0] <= al <= lo[2], couvre_haut=hi[0] <= ah <= hi[2]))
T = pd.DataFrame(out); A = T.groupby("cas").agg(vrai_bas=("vrai_bas", "first"), est_bas=("est_bas", "mean"), sd_bas=("est_bas", "std"),
                                                vrai_haut=("vrai_haut", "first"), est_haut=("est_haut", "mean"), sd_haut=("est_haut", "std"),
                                                couverture_bas=("couvre_bas", "mean"), couverture_haut=("couvre_haut", "mean"))
print(A.round(2).to_string())
A.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ecc_injection_{REL}.csv"))

# 2. ψ indépendant de β ; médiane de ṽ dépendante de β
rows = []
for a in (1.0, 1.5, 2.5):
    for b in (1.0, 1.6, 2.0):
        rs, vt, _ = ef._sky_rv(400000, a, rng); rho = np.linalg.norm(rs, axis=1); w = rho**(b - 1); w /= w.sum()
        i = rng.choice(len(rho), 150000, p=w)
        c = np.abs((rs[i]*vt[i]).sum(1))/np.linalg.norm(rs[i], axis=1)/np.linalg.norm(vt[i], axis=1)
        rows.append(dict(alpha=a, beta=b, mediane_vt=float(np.median(np.linalg.norm(vt[i], axis=1))),
                         psi_moyen=float(np.degrees(np.arccos(np.clip(c, 0, 1))).mean())))
B = pd.DataFrame(rows); print(B.round(3).to_string(index=False))
B.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta_effect.csv"), index=False)
