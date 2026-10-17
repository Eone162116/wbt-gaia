import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # racine du dépôt
# Étude v0.6, à lancer depuis la racine du dépôt après la répétition DR3 : python validation_contamination/quantile_study.py
import numpy as np, pandas as pd, json
from wbt_pipeline import select, hr_band, MG_SUN
from calibrate import kepler_geometry, simulate
import contamination as ct
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]
K = pd.read_pickle("K_dr3.pkl"); b, s, _ = hr_band(K)
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (b, s, P["hr_height"]), **kw); S = K[m]
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
dmod = 5*np.log10(S.d.values/10)
geo = dict(x=S.x.values, vc=S.vc.values, sv=(S.svt*S.vc).values, d=S.d.values, rsky=S.rsky.values, M=S.M.values,
           m1=mass(S.MG1.values), m2=mass(S.MG2.values), G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
def stepq(x, vt, q):
    lo, hi = vt[x < 0.3], vt[x > 1.0]
    return np.quantile(hi, q)/np.quantile(lo, q) - 1
QS = [0.25, 0.375, 0.5]
rng = np.random.default_rng(5); R = 300; rows = []
def run(label, gen):
    res = np.array([[stepq(*xv, q) for q in QS] for xv in (gen() for _ in range(R))])
    for j, q in enumerate(QS): rows.append(dict(cas=label, quantile=q, S_moy=res[:, j].mean(), S_sd=res[:, j].std()))
run("Newton pur", lambda: simulate(geo["x"], geo["vc"], geo["sv"], "newton", rng))
run("MOND pur", lambda: simulate(geo["x"], geo["vc"], geo["sv"], "mond", rng))
for f in (0.376, 0.5):
    cfg = dict(q_range=[0.05, 1.0], stability=0.2, shape="raghavan2010", screen="epoques+image_pess")
    run(f"Newton contaminé f={f}", lambda: ct.contaminated_newton(geo, f, rng, cfg, P["vtmax"], P["dvsky"], kepler_geometry))
T = pd.DataFrame(rows).pivot(index="cas", columns="quantile", values=["S_moy", "S_sd"]).round(4)
print(T.to_string()); pd.DataFrame(rows).to_csv("validation_contamination/quantile_study.csv", index=False)
