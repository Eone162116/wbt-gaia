import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # racine du dépôt
# Étude v0.6, à lancer depuis la racine du dépôt après la répétition DR3 : python validation_contamination/power_v06.py
import numpy as np, pandas as pd, json
from scipy.stats import norm
from wbt_pipeline import select, hr_band, MG_SUN
from calibrate import kepler_geometry, simulate, step_stat
import contamination as ct
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]
K = pd.read_pickle("K_dr3.pkl"); b, s, _ = hr_band(K)
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (b, s, P["hr_height"]), **kw); S = K[m]; N0 = len(S)
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
def geo_at(scale):
    d = S.d.values*scale; dmod = 5*np.log10(d/10)
    return dict(x=S.x.values, vc=S.vc.values, sv=(S.svt*S.vc).values, d=d, rsky=S.rsky.values, M=S.M.values,
                m1=mass(S.MG1.values), m2=mass(S.MG2.values), G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
rng = np.random.default_rng(21); R = 300
cfg = dict(q_range=[0.05, 1.0], stability=0.2, shape="raghavan2010", screen="epoques+image_pess")
out = {}
for scale in (1.0, 1.33):
    g = geo_at(scale)
    for f in (0.376, 0.5):
        v = np.array([step_stat(*ct.contaminated_newton(g, f, rng, cfg, P["vtmax"], P["dvsky"], kepler_geometry)) for _ in range(R)])
        out[(scale, f)] = (np.nanmean(v), np.nanstd(v))
    print(f"distances ×{scale}: f=0,376 → S {out[(scale,0.376)][0]:+.3f} ± {out[(scale,0.376)][1]:.3f} ; f=0,5 → S {out[(scale,0.5)][0]:+.3f} ± {out[(scale,0.5)][1]:.3f}")
mk = pd.read_csv("mocks_dr3.csv"); muN, sdN, muM, sdM = mk.newton.mean(), mk.newton.std(), mk.mond.mean(), mk.mond.std()
z = norm.isf(0.00135)
def power(N, env, truth=None):
    sc = np.sqrt(N0/N)
    thrN = max([muN + z*sdN*sc] + [mu + z*sd*sc for mu, sd in env])
    thrM = muM - z*sdM*sc
    mu3, sd3 = truth or out[(1.0, 0.376)]  # vérité : Newton contaminé à la borne haute de la bande mesurée (même vérité pour les deux enveloppes)
    return norm.cdf(thrM, mu3, sd3*sc), norm.sf(thrN, muM, sdM*sc)
print("\n      N   | ancienne enveloppe (f ≤ 0,5)        | enveloppe mesurée (f ≤ 0,38)")
print("          | MOND rej.|Newton  Newton rej.|MOND  | MOND rej.|Newton  Newton rej.|MOND")
for N in (2500, 5000, 10000):
    a = power(N, [out[(1.0, 0.5)]]); b2 = power(N, [out[(1.0, 0.376)]])
    print(f"  {N:6d}  |   {a[0]:.2f}            {a[1]:.2f}            |   {b2[0]:.2f}            {b2[1]:.2f}")
print("\nExtension à ~200 pc (N ≈ 1,9 × N150, biais recalculé à distances ×1,33), enveloppe mesurée :")
for N150 in (2500, 3000):
    N200 = int(1.9*N150); b3 = power(N200, [out[(1.33, 0.376)]], truth=out[(1.33, 0.376)])
    print(f"  N150={N150} → N200≈{N200} : MOND rejeté|Newton {b3[0]:.2f} ; Newton rejeté|MOND {b3[1]:.2f}")
