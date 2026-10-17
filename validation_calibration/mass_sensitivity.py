"""v0.7-E — Sensibilité de S aux erreurs de masse (audit externe, point 9), par simulation seulement.
Avec des masses photométriques erronées, x = r_sky/r_M et ṽ dépendent tous deux de M^−½ : un biais de masse uniforme
s'annule dans S, mais pas un biais corrélé à la couleur (donc à x). Simulations newtoniennes sur la géométrie de
l'échantillon principal : M_vraie = Σ m_phot,i × e_i, ṽ_obs = ṽ_képlérien × (M_vraie/M_phot)^½, binage sur x photométrique.
À lancer depuis la racine du dépôt après « build » : python validation_calibration/mass_sensitivity.py [dr3|dr4]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from wbt_pipeline import select, hr_band, MG_SUN, observe
import calibrate as cal

REL = sys.argv[1] if len(sys.argv) > 1 else "dr3"
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]
K = pd.read_pickle(f"K_{REL}.pkl"); hb, hs, _ = hr_band(K)
if P.get("hr_intercept") is not None: hb = P["hr_intercept"]
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (hb, hs, P["hr_height"]), **kw); S = K[m]
x, vc, rsky = S.x.values, S.vc.values, S.rsky.values
sv = np.stack([S.cv_aa, S.cv_ad, S.cv_dd, S.fd], 1)
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
m1, m2, c1, c2 = mass(S.MG1.values), mass(S.MG2.values), S.c1.values, S.c2.values
cmed = np.median(np.r_[c1, c2]); R = int(os.environ.get("WBT_STUDY_R", "2000")); rng = np.random.default_rng(3)

def run(err1, err2):
    out = []
    for _ in range(R):
        e1, e2 = err1(), err2()
        rho = (m1*e1 + m2*e2)/(m1 + m2)                      # M_vraie / M_phot
        vt_vec, _ = cal.kepler_geometry(len(x), rng)
        v = observe(vt_vec*(vc*np.sqrt(rho))[:, None], sv, rng)
        va = np.linalg.norm(v, axis=1); vt = va/vc; keep = (vt < P["vtmax"]) & (va < P["dvsky"])
        out.append(cal.step_stat(x[keep], vt[keep]))
    return np.nanmean(out), np.nanstd(out)/np.sqrt(R)

n = len(x); one = lambda: np.ones(n)
cases = {
    "aucune erreur": (one, one),
    "biais uniforme +20 %": (lambda: np.full(n, 1.2), lambda: np.full(n, 1.2)),
    "dispersion 10 % par étoile": (lambda: np.exp(rng.normal(0, 0.10, n)), lambda: np.exp(rng.normal(0, 0.10, n))),
    "dispersion 20 % par étoile": (lambda: np.exp(rng.normal(0, 0.20, n)), lambda: np.exp(rng.normal(0, 0.20, n))),
    "biais +10 %/mag vers le rouge": (lambda: 1 + 0.10*(c1 - cmed), lambda: 1 + 0.10*(c2 - cmed)),
    "biais −10 %/mag vers le rouge": (lambda: 1 - 0.10*(c1 - cmed), lambda: 1 - 0.10*(c2 - cmed)),
}
rows = []
for name, (a, b) in cases.items():
    mu, err = run(a, b); rows.append(dict(cas=name, S_newton=mu, erreur_MC=err)); print(f"{name:32s} S = {mu:+.4f} ± {err:.4f}", flush=True)
T = pd.DataFrame(rows); T["biais"] = T.S_newton - T.S_newton.iloc[0]
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"mass_sensitivity_{REL}.csv"), index=False)
print(f"corrélation couleur moyenne – x : {np.corrcoef(0.5*(c1 + c2), np.log(x))[0, 1]:+.2f}")
print(T.round(4).to_string(index=False))
