import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # racine du dépôt
# Étude v0.6, à lancer depuis la racine du dépôt après la répétition DR3 : python validation_contamination/envelope_study.py
import numpy as np, pandas as pd, json
from wbt_pipeline import select, hr_band, MG_SUN
from calibrate import kepler_geometry, step_stat
import contamination as ct
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]
K = pd.read_pickle("K_dr3.pkl"); b, s, _ = hr_band(K)
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (b, s, P["hr_height"]), **kw); S = K[m]
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
dmod = 5*np.log10(S.d.values/10)
geo = dict(x=S.x.values, vc=S.vc.values, sv=(S.svt*S.vc).values, d=S.d.values, rsky=S.rsky.values, M=S.M.values,
           m1=mass(S.MG1.values), m2=mass(S.MG2.values), G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
base = dict(q_range=[0.05, 1.0], stability=0.2)
scen = [("raghavan2010", f) for f in (0.0, 0.226, 0.30, 0.376, 0.5)] + [("mdwarf_pokemon", f) for f in (0.141, 0.19, 0.234)]
rng = np.random.default_rng(99); R = 300; rows = []
for screen in ("dr3_ruwe+image_pess", "epoques+image_pess"):
    for shape, f in scen:
        cfg = dict(base, shape=shape, screen=screen)
        Sv = np.array([step_stat(*ct.contaminated_newton(geo, f, rng, cfg, P["vtmax"], P["dvsky"], kepler_geometry)) for _ in range(R)])
        rows.append(dict(crible=screen, forme=shape, f_in=f, S_moy=np.nanmean(Sv), S_sd=np.nanstd(Sv)))
T = pd.DataFrame(rows); T.to_csv("validation_contamination/envelope_study.csv", index=False)
pd.set_option("display.width", 200); print(T.round(4).to_string(index=False))
