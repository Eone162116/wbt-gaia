"""Multivers de spécifications du WBT sur Gaia DR3 + figures.
Usage : python run_multiverse.py   (nécessite K.pkl produit par wbt_pipeline : voir README)"""
import itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wbt_pipeline import *

K = pd.read_pickle("K.pkl")
b, slope, _ = hr_band(K)

FACTORS = dict(
    degroup=[None, "snr40", "snr10"],
    hr=[None, 1.2, 0.8],            # None = pas de filtre HR ; sinon hauteur de bande (mag)
    sph=[False, True],
    ipd=[False, True],
    dmax=[150, 130],
    ruwe=[1.4, 1.25],
)
rows = []
for combo in itertools.product(*FACTORS.values()):
    spec = dict(zip(FACTORS.keys(), combo))
    hrh = spec.pop("hr")
    m, vt = select(K, (b, slope, hrh if hrh else 0.8), hr=hrh is not None, **spec)
    B, s = binned_test(K.x.values[m], vt[m])
    last = B.med.iloc[-2:].mean() if len(B) >= 4 else np.nan
    first = B.med.iloc[:2].mean() if len(B) >= 4 else np.nan
    rows.append(dict(**spec, hr=hrh, n=s["n"], chi2_N=s["chi2_N"], chi2_M=s["chi2_M"],
                     log10_pref_newton=np.log10(s["pref_newton"]), step=last/first - 1))
R = pd.DataFrame(rows).sort_values("log10_pref_newton").reset_index(drop=True)
R.to_csv("multiverse_results.csv", index=False)

print(f"{len(R)} spécifications ; favorables à MOND : {(R.log10_pref_newton < 0).sum()}")
print("médiane log10(Newton/MOND) =", round(R.log10_pref_newton.median(), 2))
eff = []
for f, levels in FACTORS.items():
    for L in levels:
        sel = R[f].isna() if L is None else (R[f] == L)
        eff.append(dict(facteur=f, niveau=str(L), med_log10=R.loc[sel, "log10_pref_newton"].median(),
                        frac_mond=(R.loc[sel, "log10_pref_newton"] < 0).mean(), n_paires_med=R.loc[sel, "n"].median()))
E = pd.DataFrame(eff); E.to_csv("multiverse_factor_effects.csv", index=False)
print(E.round(2).to_string(index=False))

# ---------------------------------------------------------------- figure 1 : courbe de spécification
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.6]))
xs = np.arange(len(R))
col = np.where(R.log10_pref_newton < 0, "#DD8452", "#4C72B0")
a1.scatter(xs, R.log10_pref_newton, c=col, s=14)
a1.axhline(0, color="k", lw=0.8)
a1.set_ylabel("log10 (vraisemblance Newton / MOND)")
nm = (R.log10_pref_newton < 0).sum()
nd = ((R.log10_pref_newton < 0) & R.degroup.isna()).sum()
a1.set_title(f"Multivers WBT Gaia DR3 : {nm}/{len(R)} spécifications favorisent MOND, dont {nd} sans dégroupage",
             loc="left", fontweight="bold")
a1.text(0.01, 0.95, "orange : MOND préféré   bleu : Newton préféré", transform=a1.transAxes, va="top", fontsize=9)
labels, ylab = [], []
lines = [("Dégroupage absent", R.degroup.isna()), ("Dégroupage S/N>40", R.degroup == "snr40"),
         ("Dégroupage S/N>10", R.degroup == "snr10"), ("Filtre HR absent", R.hr.isna()),
         ("Correction sphérique", R.sph), ("ipd_frac_multi_peak = 0", R.ipd),
         ("d < 130 pc", R.dmax == 130), ("RUWE < 1.25", R.ruwe == 1.25)]
for k, (lab, sel) in enumerate(lines):
    a2.scatter(xs[sel.values], np.full(sel.sum(), k), marker="|", s=40, color="#333333")
    ylab.append(lab)
a2.set_yticks(range(len(lines))); a2.set_yticklabels(ylab, fontsize=9); a2.invert_yaxis()
a2.set_xlabel("Spécifications classées par préférence")
for a in (a1, a2):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.text(0.01, 0.005, "Source : Gaia DR3, 1 426 271 étoiles (parallaxe > 6 mas, S/N > 10). Statistique : médiane de ṽ par bin de r/rM, χ² Newton plat vs MOND (Cookson+2026 éq. 7).",
         fontsize=8, color="#555555")
plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig("fig_specification_curve.png", dpi=150)

# ---------------------------------------------------------------- figure 2 : baseline vs sans dégroupage
fig, ax = plt.subplots(figsize=(9, 5.5))
for name, kw, c, mk in [("Checklist complète (dégroupage S/N>10)", {}, "#4C72B0", "o"),
                        ("Même chose sans dégroupage", dict(degroup=None), "#DD8452", "s")]:
    m, vt = select(K, (b, slope, 0.8), **kw)
    B, s = binned_test(K.x.values[m], vt[m])
    ax.errorbar(B.xmed, B.med, yerr=B.err, fmt=mk, color=c, capsize=3,
                label=f"{name} — N={s['n']}, Newton/MOND = {s['pref_newton']:.3g}")
    if not kw:
        xx = np.logspace(-1.3, 0.8, 200)
        ax.plot(xx, np.full_like(xx, s["A_N"]), color="#4C72B0", lw=1, label="Newton (plat)")
        Bn = B
        ax.plot(xx, (Bn.mond / mond_boost(Bn.xmed)).iloc[0] * mond_boost(xx), "--", color="#555555", lw=1,
                label="MOND (marche de +17 %)")
ax.set_xscale("log"); ax.set_xlabel("r_sky / r_M (séparation projetée / rayon MOND)")
ax.set_ylabel("ṽ médian (vitesse / vitesse circulaire newtonienne)")
ax.set_title("Le dégroupage seul fait apparaître ou disparaître la marche « MOND »", loc="left", fontweight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); plt.savefig("fig_baseline_vs_nodegroup.png", dpi=150)
print("figures écrites")
