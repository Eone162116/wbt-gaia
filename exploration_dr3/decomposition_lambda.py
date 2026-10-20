"""EXPLORATOIRE (après la levée DR3 et la clôture du test conjoint ; aucune décision n'en dépend).
D'où vient le Λ observé sur DR3 (+4,0, répétition v0.9.4) ? Décomposition descriptive de Λ paire par paire au point profilé de
chaque hypothèse, par zone et par tranche de log ṽ, comparée à ce qu'en attendent Newton central et MOND central (jeux simulés à la
même géométrie, tirés conditionnellement à la sélection depuis la v0.9.9 : N simulé = N observé). Mêmes gabarits que la répétition (graine C["seed"] + 9, R de la configuration) : Λ observé est d'abord reproduit.
À lancer depuis la racine après « blind » et « unblind » DR3 : python exploration_dr3/decomposition_lambda.py"""
import os, sys, json, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "validation_joint"))
import numpy as np, pandas as pd
import jointlik as jl
from common import load_inputs, provenance
from wbt_pipeline import select, hr_band
t0 = time.time(); REL = "dr3"
CFG = json.load(open("config_prereg.json", encoding="utf-8")); C = CFG["calibration"]; ST = CFG["statistic"]
I = load_inputs(REL); P, CC, GRID, R, G, zs, a_c, b_c, f = (I[k] for k in ("P", "CC", "GRID", "R", "G", "zs", "a_c", "b_c", "f"))
fb = {sh: f[sh] for sh in f}; sh0 = "raghavan2010"
cont = [(sh0, 0.0)] + [(sh, fb[sh][1]) for sh in fb] + [(sh0, fb[sh0][2])]
T = jl.templates(G, zs, b_c, cont, CC, np.random.default_rng(C["seed"] + 9), R, P["vtmax"], P["dvsky"], GRID)
K = pd.read_pickle(f"K_{REL}.pkl"); hb, hs, _ = hr_band(K)
if P.get("hr_intercept") is not None: hb = P["hr_intercept"]
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, vt = select(K, (hb, hs, P["hr_height"]), **kw)
lv_o, ps_o, k_o = np.log10(vt[m]), K.psi.values[m], np.ones(int(m.sum()), bool)
OUT = []
def say(s): print(s, flush=True); OUT.append(s)
lam_o = jl.lam(lv_o, ps_o, k_o, zs, T)
R0 = json.load(open(f"RESULT_{REL}.json", encoding="utf-8")).get("conjoint", {}).get("lam_obs") if os.path.exists(f"RESULT_{REL}.json") else None
say(f"Λ observé recalculé = {lam_o:+.3f} (RESULT_{REL}.json : {R0:+.3f})" if R0 is not None else f"Λ observé recalculé = {lam_o:+.3f}")

def contrib(parts):
    sums = [p.sum(-1, dtype=np.float64) for p, _ in parts]; out = []
    for g in (0, 1):
        llz = [s[g].max(1) for s in sums]; tot = sum(llz); ci, ei = np.unravel_index(np.argmax(tot), tot.shape)
        out.append(np.concatenate([p[g, ci, np.argmax(s[g][ci, :, ei]), ei] for (p, _), s in zip(parts, sums)]))
    return 2*(out[1] - out[0]), np.concatenate([ii for _, ii in parts])
BINS = [-9, -0.6, -0.3, 0.0, 0.15, 0.25, 9]; LAB = ["<-0,6", "-0,6..-0,3", "-0,3..0", "0..0,15", "0,15..0,25", ">0,25"]
def by_bin(lv, ps, k):
    parts = jl.lam_parts(lv, ps, k, zs, T); c_, ii = contrib(parts); hz = np.isin(ii, zs["haut"]); b = np.digitize(lv[ii], BINS) - 1
    return (np.array([c_[hz & (b == q)].sum() for q in range(len(LAB))] + [c_[~hz].sum()]),
            np.array([(hz & (b == q)).sum() for q in range(len(LAB))] + [(~hz).sum()]))
obs_c, obs_n = by_bin(lv_o, ps_o, k_o)
rng = np.random.default_rng(4343); NM = 100; sims = {}
for g, name in ((0, "newton"), (1, "mond")):
    cs, ns = [], []
    for _ in range(NM):
        lv, ps, k = jl.simulate_obs_cond(G, a_c, b_c, g, f["raghavan2010"][1], "raghavan2010", CC, rng, P["vtmax"], P["dvsky"], grid=GRID)
        c_, n_ = by_bin(lv, ps, k); cs.append(c_); ns.append(n_)
    sims[name] = (np.array(cs), np.array(ns))
rows = []
for q, lab in enumerate(LAB + ["zone basse (toute)"]):
    cN, cM = sims["newton"][0][:, q], sims["mond"][0][:, q]; nN, nM = sims["newton"][1][:, q], sims["mond"][1][:, q]
    rows.append(dict(tranche=("zone haute " + lab) if q < len(LAB) else lab, n_obs=int(obs_n[q]), n_newton=round(nN.mean(), 1), n_mond=round(nM.mean(), 1),
                     contrib_par_paire_obs=round(obs_c[q]/max(obs_n[q], 1), 3),
                     contrib_par_paire_newton=round(float(np.nanmean(cN/np.where(nN > 0, nN, np.nan))), 3),
                     contrib_par_paire_mond=round(float(np.nanmean(cM/np.where(nM > 0, nM, np.nan))), 3),
                     contrib_obs=round(obs_c[q], 2), contrib_newton=round(cN.mean(), 2), contrib_mond=round(cM.mean(), 2),
                     z_vs_newton=round((obs_c[q] - cN.mean())/max(cN.std(), 1e-9), 2)))
D = pd.DataFrame(rows); out_csv = os.path.join("exploration_dr3", "decomposition_lambda_dr3.csv"); D.to_csv(out_csv, index=False)
say(D.to_string(index=False))
say(f"total : observé {obs_c.sum():+.2f} ; attendu Newton central {sims['newton'][0].sum(1).mean():+.2f} ± {sims['newton'][0].sum(1).std():.2f} ; "
    f"MOND central {sims['mond'][0].sum(1).mean():+.2f} ± {sims['mond'][0].sum(1).std():.2f} ({NM} jeux par hypothèse, f_in central)")
say(f"({time.time()-t0:.0f} s)")
open(os.path.join("exploration_dr3", "decomposition_lambda_dr3.txt"), "w", encoding="utf-8").write("\n".join(OUT) + "\n")
provenance(REL, "exploration_dr3/decomposition_lambda.py", dict(gabarits=C["seed"] + 9, jeux=4343), out_csv, R=R, n_jeux_par_hypothese=NM,
           statut="exploratoire, après levée et clôture ; aucune décision n'en dépend")
