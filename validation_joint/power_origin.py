"""v0.9.2 — Diagnostic (simulation seulement, géométrie DR3, aucune vitesse observée) : d'où vient la puissance du test conjoint v0.9.1 ?
Décomposition de Λ par paire au point profilé de chaque hypothèse, et variante tronquée à ṽ < √2 (log ṽ < 0,15) pour comparaison.
À lancer depuis la racine du dépôt après « blind » : python validation_joint/power_origin.py [dr3|dr4]. Sortie : validation_joint/power_origin_<rel>.txt"""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import builtins
OUT = []
def print(*a, **k):
    builtins.print(*a, **k); OUT.append(" ".join(str(x) for x in a))
exec(open("validation_joint/stability_study.py", encoding="utf-8").read().split("rng = np.random.default_rng(910)")[0].split('"""', 2)[2])
import jointlik as jl
t0 = time.time()
T = jl.templates(G, zs, b_c, CONT, CC, np.random.default_rng(777), R=R, vtmax=P["vtmax"], dvsky=P["dvsky"], grid=GRID)
print(f"gabarits ({time.time()-t0:.0f} s)", flush=True)
rng = np.random.default_rng(4242); fc, fh = f["raghavan2010"][1], 0.5
sets = {"newton": [jl.simulate_obs(G, a_c, b_c, 0, fc, "raghavan2010", CC, rng, P["vtmax"], P["dvsky"], grid=GRID) for _ in range(40)],
        "newton_f05": [jl.simulate_obs(G, a_c, b_c, 0, fh, "raghavan2010", CC, rng, P["vtmax"], P["dvsky"], grid=GRID) for _ in range(40)],
        "mond": [jl.simulate_obs(G, a_c, b_c, 1, fc, "raghavan2010", CC, rng, P["vtmax"], P["dvsky"], grid=GRID) for _ in range(40)]}
def contrib(parts):
    sums = [p.sum(-1, dtype=np.float64) for p, _ in parts]; out = []
    for g in (0, 1):
        llz = [s[g].max(1) for s in sums]; tot = sum(llz); ci, ei = np.unravel_index(np.argmax(tot), tot.shape)
        out.append(np.concatenate([p[g, ci, np.argmax(s[g][ci, :, ei]), ei] for (p, _), s in zip(parts, sums)]))
    return 2*(out[1] - out[0]), np.concatenate([ii for _, ii in parts])
BINS = [-9, -0.6, -0.3, 0.0, 0.15, 0.25, 9]; LAB = ["<-0,6", "-0,6..-0,3", "-0,3..0", "0..0,15", "0,15..0,25", ">0,25"]
for variant, cut in (("complet (ṽ < 2,5)", T["c"]), ("tronqué (ṽ < √2)", 0.15)):
    Tv = dict(T, c=cut); res = {}; infl = []
    for h, L in sets.items():
        lam, byb = [], []
        for j, (lv, ps, k) in enumerate(L):
            kk = k & (lv < cut); parts = jl.lam_parts(lv, ps, kk, zs, Tv); c_, ii = contrib(parts); lam.append(c_.sum())
            hz = np.isin(ii, zs["haut"]); b = np.digitize(lv[ii], BINS) - 1
            byb.append([c_[hz & (b == q)].sum() for q in range(len(LAB))] + [c_[~hz].sum()])
            if h in ("newton", "mond") and j < 10: infl.append(np.abs(jl.loo_influence(lv, ps, kk, zs, Tv, parts)[1]).max())
        res[h] = (np.array(lam), np.array(byb).mean(0))
    sN, sM = res["newton"][0].std(), res["mond"][0].std(); sd = 0.5*(sN + sM)
    sep = (res["mond"][0].mean() - res["newton"][0].mean())/sd
    print(f"\n{variant} : séparation MOND−Newton = {sep:.2f}σ (σ_N {sN:.1f}, σ_M {sM:.1f}) ; biais f_in 0,5 sous Newton = "
          f"{(res['newton_f05'][0].mean() - res['newton'][0].mean())/sN:+.2f}σ ; influence max d'une paire, 90e centile (20 jeux) = {np.quantile(infl, 0.9):.2f}", flush=True)
    d = res["mond"][1] - res["newton"][1]
    print("  part de la séparation (E_MOND − E_Newton des contributions) par tranche de log ṽ, zone haute : "
          + " ; ".join(f"{l} {v:+.1f}" for l, v in zip(LAB, d[:-1])) + f" ; zone basse {d[-1]:+.1f} ; total {d.sum():+.1f}", flush=True)
print(f"({time.time()-t0:.0f} s)")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"power_origin_{REL}.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
