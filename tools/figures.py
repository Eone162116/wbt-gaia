"""v0.9.14 — Figures et tables fixées à l'avance pour l'article (section 12 de la section « Methods »).
Usage, depuis la racine du dépôt, APRÈS la levée : python tools/figures.py dr3|dr4  →  dossier figures_<rel>/ (PDF, PNG, CSV, LaTeX).
Le script refuse de tourner tant que RESULT_<rel>.json n'atteste pas une levée : il lit les vitesses observées.
Entrées : K_<rel>.pkl, eccfit_<rel>.csv, finfer_<rel>.csv, mocks_<rel>.csv, envelope_<rel>.csv, joint_mocks_<rel>.csv,
RESULT_<rel>.json, CONTROLS_<rel>_aveugle.json, multiverse_<rel>.csv ; facultatives : multiverse_<rel>_par_axe.csv,
exploration_<rel>/decomposition_lambda_<rel>.csv, DEVIATIONS_<rel>.json. Aucune entrée n'est modifiée.
Figures 1 à 6 et tables 1 à 5 ; les tables A1 et A2 de l'annexe sont rédigées à la main."""
import os, sys, json
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wbt_pipeline import hr_band, in_hr, observe
import calibrate as cal, eccfit as ef

REL = sys.argv[1] if len(sys.argv) > 1 else "dr3"
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]; ST = CFG["statistic"]
R = json.load(open(f"RESULT_{REL}.json", encoding="utf-8")) if os.path.exists(f"RESULT_{REL}.json") else {}
if not R.get("unblinded_utc") or not R.get("freeze_manifest"):
    sys.exit(f"ARRÊT : RESULT_{REL}.json n'atteste pas une levée ; ces figures lisent les vitesses observées.")
OUT = f"figures_{REL}"; os.makedirs(OUT, exist_ok=True)
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
NEWTON, MOND, OBS, GREY = OKABE[0], OKABE[5], "#000000", "#9a9a9a"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
LABEL = REL.upper()

def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig); print(f"  {name}")

def latex(df, name, caption):
    """Table LaTeX simple (booktabs), sans dépendance externe ; le CSV jumeau garde toutes les décimales."""
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    def cell(v):
        if isinstance(v, (float, np.floating)): return "--" if not np.isfinite(v) else (f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:.3f}")
        return str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}", rf"\caption{{{caption}}}", r"\begin{tabular}{" + "l" * df.shape[1] + "}", r"\toprule",
             " & ".join(cell(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, f"{name}.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n"); print(f"  {name}")

# ---- échantillon principal, reconstruit exactement comme run_wbt.primary_selection, et étapes de sélection
K = pd.read_pickle(f"K_{REL}.pkl"); b, slope, _ = hr_band(K)
if P.get("hr_intercept") is not None: b = P["hr_intercept"]
vt = K.vt_sph if P["sph"] else K.vt_raw; v = K.v_sph if P["sph"] else K.v_raw
hrp = (b, slope, P["hr_height"])
steps = [("Mutual pairs in the catalogue", np.ones(len(K), bool)),
         (f"Common distance < {P['dmax']} pc", (K.d < P["dmax"]).values),
         (f"Separation {P['smin']/1e3:g}–{P['smax']/1e3:g} kau", ((K.rsky > P["smin"]) & (K.rsky < P["smax"])).values),
         (f"Degrouping ({P['degroup']})", ((K.nc10 == 1) if P["degroup"] == "snr10" else (K.nc40 == 1) if P["degroup"] == "snr40"
                                            else pd.Series(True, index=K.index)).values),
         (f"RUWE < {P['ruwe']}" + (", ipd_frac_multi_peak = 0" if P["ipd"] else ""),
          ((K.ruwe_max < P["ruwe"]) & ((K.ipd_max == 0) if P["ipd"] else True)).values),
         (f"Both RVs, ΔRV < {P['drv']} km/s", (K.both_rv & (K.drv < P["drv"])).values),
         (f"σ(ṽ) < {P['svt']}", (K.svt < P["svt"]).values),
         (f"ṽ < {P['vtmax']}, Δv_sky < {P['dvsky']} km/s", ((vt < P["vtmax"]) & (v < P["dvsky"])).values),
         (f"HR band (b = {b:.3f}, height {P['hr_height']} mag)", (in_hr(K.c1, K.MG1, *hrp) & in_hr(K.c2, K.MG2, *hrp)).values if P["hr"]
          else np.ones(len(K), bool))]
if REL == "dr4" and P["screen"] == "dr4" and "scr4" in K.columns: steps.append(("DR4 star screen and flags", (K.scr4 & K.flags_ok).values))
if REL == "dr4" and P["epoch"] and "epoch_ok" in K.columns: steps.append(("Epoch-astrometry screen", K.epoch_ok.values))
cum = np.ones(len(K), bool); flow = []
for lab, m in steps:
    cum = cum & np.asarray(m, bool); flow.append((lab, int(cum.sum())))
msk = cum
if int(msk.sum()) != R.get("N", int(msk.sum())):
    sys.exit(f"ARRÊT : la sélection reconstruite donne N = {int(msk.sum())}, RESULT_{REL}.json indique N = {R.get('N')}.")
S_ = K[msk].reset_index(drop=True); vt_o = vt[msk].values; x_o = S_.x.values
print(f"{LABEL} : N = {len(S_)} (reconstruit, identique au résultat levé) ; figures dans {OUT}/")

# ---- Table 1 et figure 1 : échantillon
latex(pd.DataFrame(flow, columns=["Cut (cumulative)", "Pairs remaining"]), "table1_cutflow",
      f"Selection cuts applied in order, and the number of pairs remaining ({LABEL}).")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.1], wspace=0.95))
c_all = np.r_[K.c1, K.c2]; m_all = np.r_[K.MG1, K.MG2]; c_s = np.r_[S_.c1, S_.c2]; m_s = np.r_[S_.MG1, S_.MG2]
ax[0].scatter(c_all, m_all, s=0.3, c=GREY, alpha=0.25, rasterized=True, label="all mutual pairs")
ax[0].scatter(c_s, m_s, s=1.5, c=NEWTON, rasterized=True, label="primary sample")
cc = np.linspace(0.5, 3.5, 2)
for dy, ls in ((0, "-"), (-P["hr_height"]/2, "--"), (P["hr_height"]/2, "--")): ax[0].plot(cc, b + slope*cc + dy, ls, c="k", lw=0.8)
ax[0].set(xlim=(0, 4), ylim=(16, 2), xlabel="BP − RP", ylabel="M_G", title="Colour–magnitude diagram and HR band"); ax[0].legend(markerscale=6, loc="lower left")
ax[1].scatter(K.d, K.rsky/1e3, s=0.3, c=GREY, alpha=0.2, rasterized=True); ax[1].scatter(S_.d, S_.rsky/1e3, s=2, c=NEWTON, rasterized=True)
ax[1].set(yscale="log", xlabel="Common distance (pc)", ylabel="Projected separation (kau)", title="Separation–distance plane")
ax[2].barh([f[0] for f in flow][::-1], [f[1] for f in flow][::-1], color=NEWTON); ax[2].set(xscale="log", xlabel="Pairs remaining", title="Cut flow")
ax[2].tick_params(axis="y", labelsize=7)
for k, n in enumerate([f[1] for f in flow][::-1]): ax[2].text(n*1.08, k, f"{n:,}".replace(",", " "), va="center", fontsize=6)
fig.suptitle(f"Figure 1 — Sample ({LABEL}, N = {len(S_)})"); save(fig, "fig1_sample")

# ---- Table 2 : nuisances mesurées sans la norme des vitesses
E = pd.read_csv(f"eccfit_{REL}.csv"); F = pd.read_csv(f"finfer_{REL}.csv")
t2 = [dict(Quantity=f"α ({'x < ' + str(ST['x_low']) if r.zone == 'bas' else 'x > ' + str(ST['x_high'])}, {r.gabarits} templates)" if r.gabarits != "beta_separations"
           else f"β ({'x < ' + str(ST['x_low']) if r.zone == 'bas' else 'x > ' + str(ST['x_high'])})",
           Low=r.alpha_min, Central=r.alpha, High=r.alpha_max, N=int(r.n)) for r in E.itertuples()]
t2 += [dict(Quantity=f"f_in ({r.shape}), flag rate {r.taux:.3f} ± {r.taux_err:.3f} on {int(r.n)} pairs ({r.drapeau})",
            Low=r.f_bas, Central=r.f_central, High=r.f_haut, N=int(r.n)) for r in F.itertuples()]
latex(pd.DataFrame(t2), "table2_nuisances", f"Nuisances measured without the velocity norm ({LABEL}): ±2σ bands.")

# ---- Figure 2 : médiane de ṽ par tranche de x, attentes Newton et MOND (scénario central, bande des scénarios mesurés)
def band_of(gab):
    rows = E[E.gabarits == gab].set_index("zone")
    return {z: (rows.loc[z, "alpha_min"], rows.loc[z, "alpha"], rows.loc[z, "alpha_max"]) for z in ("bas", "haut")} | \
           {f"n_{z}": int(rows.loc[z, "n"]) for z in ("bas", "haut")}
ecc_sc = ef.scenarios(band_of("newton"), band_of("mond"), band_of("beta_separations"))
edges = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 6]); mid = np.sqrt(edges[:-1]*edges[1:])
def binned_median(xx, vv):
    i = np.digitize(xx, edges) - 1
    return np.array([np.median(vv[i == k]) if (i == k).sum() >= 5 else np.nan for k in range(len(mid))])
sv_o = S_.svt.values*S_.vc.values
def sim_medians(truth, scen, n=200, seed=1):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        vt_vec, ratio = cal.kepler_geometry(len(x_o), rng, ecc_sc[scen], S_.rsky.values, x_o)
        if truth == "mond": vt_vec = vt_vec*cal.mond_boost3d(x_o*ratio)[:, None]
        vv = observe(vt_vec*S_.vc.values[:, None], np.stack([S_.cv_aa, S_.cv_ad, S_.cv_dd, S_.fd], 1), rng)
        out.append(binned_median(x_o, np.linalg.norm(vv, axis=1)/S_.vc.values))
    out = np.array(out); ok = np.isfinite(out).any(0)
    return np.where(ok, np.nanmedian(np.where(np.isfinite(out), out, 0)*0 + out, 0) if ok.all() else np.array([np.nanmedian(out[:, k]) if ok[k] else np.nan for k in range(out.shape[1])]), np.nan)
obs_med = binned_median(x_o, vt_o)
rng_b = np.random.default_rng(7); boot = np.array([binned_median(x_o[j], vt_o[j]) for j in (rng_b.integers(0, len(x_o), len(x_o)) for _ in range(300))])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for truth, col in (("newton", NEWTON), ("mond", MOND)):
    curves = np.array([sim_medians(truth, s, seed=11*k + (0 if truth == "newton" else 5)) for k, s in enumerate(("mes_central", "mes_pire_newton", "mes_pire_mond"))])
    ax.fill_between(mid, curves.min(0), curves.max(0), color=col, alpha=0.18, lw=0)
    ax.plot(mid, curves[0], "-" if truth == "newton" else "--", c=col, lw=1.5, label=f"{'Newton' if truth == 'newton' else 'MOND'} (central; band: measured scenarios)")
ax.errorbar(mid, obs_med, yerr=np.nanstd(boot, 0), fmt="o", c=OBS, ms=4, capsize=2, label=f"{LABEL} observed (bootstrap 1σ)")
for xv in (ST["x_low"], ST["x_high"]): ax.axvline(xv, c=GREY, lw=0.6, ls=":")
ax.set(xscale="log", xlabel="x = r_sky / r_M", ylabel="median ṽ", title=f"Figure 2 — Median normalised velocity ({LABEL})"); ax.legend(loc="upper left")
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
save(fig, "fig2_median_vt")

# ---- Figure 3 et table 3 : S observé face à ses distributions de référence
M = pd.read_csv(f"mocks_{REL}.csv"); EN = pd.read_csv(f"envelope_{REL}.csv"); Sobs = R["S_obs"]
SHAPE = {"raghavan2010": "solar-type", "mdwarf_pokemon": "M-dwarf"}
def nice(c):
    h, _, sc = c.partition("_"); sc = {"": "thermal", "mes_central": "measured, central", "mes_pire_newton": "measured, most favourable to Newton",
                                      "mes_pire_mond": "measured, most favourable to MOND"}.get(sc, sc)
    return f"{'Newton' if h == 'newton' else 'MOND'}: {sc}"
dists = [(nice(c), M[c].dropna().values, NEWTON) for c in M.columns if c.startswith("newton")]
dists += [(f"Newton: contamination, {SHAPE.get(sh, sh)}, f_in = {f:.2f}", g.S.values, NEWTON) for (sh, f), g in EN.groupby(["shape", "f_in"])]
dists += [(nice(c), M[c].dropna().values, MOND) for c in M.columns if c.startswith("mond")]
fig, ax = plt.subplots(figsize=(6.2, 0.32*len(dists) + 1.2))
for k, (lab, d, col) in enumerate(dists):
    ax.violinplot(d, positions=[k], orientation="horizontal", widths=0.8, showextrema=False)["bodies"][0].set(facecolor=col, alpha=0.45)
    q = np.quantile(d, 1 - 0.00135) if lab.startswith("Newton") else np.quantile(d, 0.00135)
    ax.plot([q, q], [k - 0.35, k + 0.35], c=col, lw=1.2)
ax.axvline(Sobs, c=OBS, lw=1.4, label=f"observed S = {Sobs:+.3f}")
ax.set_yticks(range(len(dists))); ax.set_yticklabels([d[0] for d in dists], fontsize=7)
ax.set(xlabel="S", title=f"Figure 3 — S against its reference distributions ({LABEL}); ticks: one-sided 3σ quantiles")
ax.legend(loc="lower right"); save(fig, "fig3_S_distributions")
rows3 = [dict(Statistic="S", Hypothesis=s["hyp"], Scenario=s["scenario"], N_sim=s["N"], p=s["p"], p_CP95=s["p_cp95"], p_Gauss=s["p_gauss"])
         for s in R["scenarios"]]
if "conjoint" in R:
    rows3 += [dict(Statistic="Λ", Hypothesis=s["hyp"], Scenario=s["scenario"], N_sim=s["N"], p=s["p"], p_CP95=s["p_cp95"], p_Gauss=s["p_gauss"])
              for s in R["conjoint"]["scenarios"]]
latex(pd.DataFrame(rows3), "table3_pvalues",
      f"p-values in every scenario ({LABEL}). Verdict of S: {R['verdict']}" + (f"; verdict of Λ (secondary): {R['conjoint']['verdict']}." if "conjoint" in R else "."))

# ---- Figure 4 et table 4 : multivers
MV = pd.read_csv(f"multiverse_{REL}.csv").sort_values("S").reset_index(drop=True)
is_prim = np.ones(len(MV), bool)
for k, val in (("degroup", P["degroup"]), ("hr", P["hr_height"] if P["hr"] else np.nan), ("sph", P["sph"]), ("dmax", P["dmax"]), ("ruwe", P["ruwe"]))\
               + ((("screen", P["screen"]), ("epoch", P["epoch"])) if REL == "dr4" else ()):
    if k in MV.columns:
        col = MV[k]; is_prim &= (col.isna().values if (isinstance(val, float) and np.isnan(val)) else (col.map(lambda u: str(u)) == str(val)).values)
if "hr_ordonnee" in MV.columns: is_prim &= (MV.hr_ordonnee == "mediane").values
axes_shown = [a for a in ("hr_ordonnee", "degroup", "dmax", "sph") if a in MV.columns]
lv = lambda a: MV[a].map(lambda u: "none" if pd.isna(u) else str(u))
fig, ax = plt.subplots(1 + len(axes_shown), 1, figsize=(7.5, 2.6 + 0.55*sum(lv(a).nunique() for a in axes_shown)),
                       sharex=True, gridspec_kw=dict(height_ratios=[3] + [0.35*lv(a).nunique() for a in axes_shown]))
ax = np.atleast_1d(ax); i = np.arange(len(MV))
ax[0].scatter(i, MV.S, s=4, c=GREY); ax[0].scatter(i[is_prim], MV.S[is_prim], s=40, c=MOND, zorder=3, label="primary specification")
ax[0].axhline(0, c=GREY, lw=0.6); ax[0].set(ylabel="S", title=f"Figure 4 — Specification curve ({LABEL}, {len(MV)} specifications)"); ax[0].legend()
for a_, axis in zip(ax[1:], axes_shown):
    order = ["mediane-0.2", "mediane-0.1", "mediane", "mediane+0.1", "mediane+0.2", "cookson", "none", "snr40", "snr10"]
    levs = sorted(lv(axis).unique(), key=lambda u: (order.index(u) if u in order else 99, u))
    for j, lev in enumerate(levs):
        on = (lv(axis) == lev).values; a_.scatter(i[on], np.full(on.sum(), j), s=2, marker="|", c=NEWTON)
    a_.set_yticks(range(len(levs))); a_.set_yticklabels([u.replace("mediane", "median") for u in levs], fontsize=6)
    a_.set_ylabel({"hr_ordonnee": "HR intercept", "degroup": "Degrouping", "dmax": "d_max (pc)", "sph": "Perspective corr."}.get(axis, axis),
                  fontsize=7, rotation=0, ha="right")
ax[-1].set_xlabel("specifications ranked by S"); save(fig, "fig4_specification_curve")
pa = f"multiverse_{REL}_par_axe.csv"
if os.path.exists(pa): T4 = pd.read_csv(pa)
else:
    import multivers as mv; T4 = mv.by_axis(MV, [a for a in CFG["multiverse"] if a in MV.columns])
latex(T4, "table4_multiverse_by_axis", f"S summarised along each axis of the multiverse ({LABEL}); descriptive, not a probability.")

# ---- Figure 5 : Λ face à ses distributions de référence, et décomposition exploratoire
if "conjoint" in R and os.path.exists(f"joint_mocks_{REL}.csv"):
    JM = pd.read_csv(f"joint_mocks_{REL}.csv"); dec = f"exploration_{REL}/decomposition_lambda_{REL}.csv"
    fig, ax = plt.subplots(1, 2 if os.path.exists(dec) else 1, figsize=(10.5 if os.path.exists(dec) else 5.5, 3.8)); ax = np.atleast_1d(ax)
    for k, ((h, sc), g) in enumerate(JM.groupby(["hyp", "scenario"])):
        ax[0].hist(g.lam, bins=60, histtype="step", lw=1.2, color=NEWTON if h == "newton" else MOND, ls=["-", "--", ":"][k % 3], label=f"{h} / {sc}")
    ax[0].axvline(R["conjoint"]["lam_obs"], c=OBS, lw=1.4, label=f"observed Λ = {R['conjoint']['lam_obs']:+.2f}")
    ax[0].set(xlabel="Λ", ylabel="simulated data sets", title="Λ against its reference distributions (secondary)"); ax[0].legend(fontsize=6)
    if os.path.exists(dec):
        D = pd.read_csv(dec); y = np.arange(len(D)); w = 0.27
        ax[1].barh(y - w, D.contrib_newton, w, color=NEWTON, label="expected, Newton"); ax[1].barh(y, D.contrib_obs, w, color=OBS, label="observed")
        ax[1].barh(y + w, D.contrib_mond, w, color=MOND, label="expected, MOND")
        ren = lambda t: ("x < 0.3 (all)" if "basse" in t else "x > 1, log ṽ " + t.replace("zone haute ", "").replace(",", ".").replace("..", " to "))
        ax[1].set_yticks(y); ax[1].set_yticklabels([ren(t) for t in D.tranche], fontsize=7); ax[1].axvline(0, c=GREY, lw=0.6)
        ax[1].set(xlabel="contribution to Λ", title="Pair-by-pair decomposition (exploratory)"); ax[1].legend(fontsize=7)
    fig.suptitle(f"Figure 5 — Secondary statistic Λ ({LABEL})"); save(fig, "fig5_lambda")

# ---- Figure 6 : contamination
CT = json.load(open(f"CONTROLS_{REL}_aveugle.json", encoding="utf-8")) if os.path.exists(f"CONTROLS_{REL}_aveugle.json") else {}
comp = CT.get("f_in_comparaison"); two = isinstance(comp, dict)
fig, ax = plt.subplots(1, 2 if two else 1, figsize=(9 if two else 5, 2.8)); ax = np.atleast_1d(ax)
for k, r in enumerate(F.itertuples()):
    ax[0].plot([r.f_bas, r.f_haut], [k, k], lw=6, c=OKABE[k], alpha=0.6, solid_capstyle="butt"); ax[0].plot(r.f_central, k, "o", c=OKABE[k])
ax[0].set_ylim(-0.6, len(F) - 0.4); ax[0].set_yticks(range(len(F))); ax[0].set_yticklabels([{"raghavan2010": "solar-type", "mdwarf_pokemon": "M-dwarf"}.get(u, u) for u in F["shape"]])
ax[0].set(xlabel="f_in (per component)",
    title=f"Inferred f_in bands (flag rate {F.taux.iloc[0]:.3f} ± {F.taux_err.iloc[0]:.3f}, {F.drapeau.iloc[0]})")
if two:
    shapes = list(comp["epoques"]); nm = {"raghavan2010": "solar-type", "mdwarf_pokemon": "M-dwarf"}
    for k, sh in enumerate(shapes):
        for j, (lab, key, col) in enumerate((("epoch screen", "epoques", OKABE[2]), ("RUWE", "ruwe_grille_dr3_approchee", OKABE[3]))):
            lo, ce, hi = comp[key][sh]; yy = k + (j - 0.5)*0.3
            ax[1].plot([lo, hi], [yy, yy], lw=5, c=col, alpha=0.6, solid_capstyle="butt", label=lab if k == 0 else None); ax[1].plot(ce, yy, "o", c=col, ms=4)
    ax[1].set_yticks(range(len(shapes))); ax[1].set_yticklabels([nm.get(u, u) for u in shapes]); ax[1].set_ylim(-0.6, len(shapes) - 0.4)
    ax[1].set(xlabel="f_in (per component)", title=f"Epoch screen vs RUWE (overlap: {comp['recouvrement']})"); ax[1].legend(fontsize=7)
fig.suptitle(f"Figure 6 — Contamination ({LABEL})", y=1.08); save(fig, "fig6_contamination")

# ---- Table 5 : écarts au pré-enregistrement
DV = json.load(open(f"DEVIATIONS_{REL}.json", encoding="utf-8")) if os.path.exists(f"DEVIATIONS_{REL}.json") else []
latex(pd.DataFrame(DV, columns=["Date", "Deviation", "Justification"]), "table5_deviations",
      f"Deviations from the pre-registration ({LABEL}); empty if none.")
print("terminé : 6 figures (PDF, PNG) et 5 tables (CSV, LaTeX).")
