"""v0.9.9 — Les jeux simulés conditionnés gardent toutes les paires de la géométrie observée (N simulé = N observé), alors que
le tirage non conditionné en perd par le crible de contamination. Géométrie synthétique, aucune donnée réelle.
Lancer depuis la racine : python tests/test_conditional.py"""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np
import jointlik as jl, contamination as ct, calibrate as cal
CFG = json.load(open("config_prereg.json", encoding="utf-8")); CC = dict(CFG["calibration"]["contamination"]); CC["screen"] = "dr3_ruwe+image_pess"
r = np.random.default_rng(1); N = 400
x = 10**r.uniform(np.log10(0.05), np.log10(4), N); d = r.uniform(40, 150, N); M = r.uniform(0.8, 1.8, N)
m1 = M*r.uniform(0.5, 0.7, N); m2 = M - m1; vc = r.uniform(0.25, 0.9, N); s = (0.05*vc)**2
G = dict(x=x, vc=vc, sv=np.stack([s, 0*s, s, np.full(N, 0.01)], 1), d=d, rsky=x*7000*np.sqrt(M), M=M, m1=m1, m2=m2,
         G1=7.5 - 6*np.log10(m1) + 5*np.log10(d/10), G2=7.5 - 6*np.log10(m2) + 5*np.log10(d/10))
args = ((2.0, 0.5), (1.4, 2.1), 0, 0.30, "raghavan2010", CC)
_, _, k0 = jl.simulate_obs(G, *args, np.random.default_rng(2), 2.5, 10.0)
lv, ps, k1 = jl.simulate_obs_cond(G, *args, np.random.default_rng(2), 2.5, 10.0)
assert k0.mean() < 0.95, f"le tirage non conditionné devrait perdre des paires (garde {k0.mean():.2f})"
assert k1.mean() >= 0.999 and np.isfinite(lv[k1]).all() and (lv[k1] < np.log10(2.5)).all(), f"tirage conditionné incomplet ({k1.mean():.4f})"
kgg = lambda n, rr, g: cal.kepler_geometry(n, rr, "thermal", g.get("rsky"), g["x"])
xx0, _ = ct.contaminated_newton(G, 0.30, np.random.default_rng(3), dict(CC, shape="raghavan2010"), 2.5, 10.0, lambda n, rr: cal.kepler_geometry(n, rr))
xx1, vv1 = ct.contaminated_newton_cond(G, 0.30, np.random.default_rng(3), dict(CC, shape="raghavan2010"), 2.5, 10.0, kgg)
assert len(xx0) < 0.95*N and len(xx1) >= 0.999*N and (vv1 < 2.5).all(), f"enveloppe : {len(xx0)} puis {len(xx1)} paires sur {N}"
st = {}
jl.simulate_obs_cond(G, *args, np.random.default_rng(4), 2.5, 10.0, stats=st)
assert st["paires"] == N and st["manquantes"] <= 1, f"comptage de complétude incorrect : {st}"
assert jl.check_completeness(st, 0.001, "test") <= 0.001
st0 = {}; _, _, k_imp = jl.simulate_obs_cond(G, *args, np.random.default_rng(5), 2.5, 0.0, max_iter=5, stats=st0)   # coupe impossible
assert not k_imp.any() and st0["manquantes"] == N, "une coupe impossible doit laisser toutes les paires manquantes"
try: jl.check_completeness(st0, 0.001, "test"); raise AssertionError("calibration incomplète non refusée (test conjoint)")
except SystemExit: pass
try:
    cal.calibrate_envelope(G, n_per_f=1, cfg=dict(CC, shape="raghavan2010"), vtmax=2.5, dvsky=0.0, scenarios=[("raghavan2010", 0.3)], max_iter=5)
    raise AssertionError("calibration incomplète non refusée (enveloppe)")
except SystemExit: pass
print(f"tests du tirage conditionné : OK, 6 cas (test conjoint {k0.mean():.2f} → {k1.mean():.3f} ; enveloppe {len(xx0)/N:.2f} → {len(xx1)/N:.3f} ; règle de complétude appliquée)")
