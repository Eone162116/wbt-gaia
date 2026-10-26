"""v0.9 — Critère de promotion du test conjoint (règle fixée le 18/10/2026, avant tout calcul de Λ sur données réelles) :
(a) gain de probabilité d'un verdict tranché à ≈ 2400 paires ≥ +0,10 par rapport au test S (règle v0.8.1) ;
(b) biais de Λ sous Newton < 0,5σ pour chaque test de robustesse non couvert par les gabarits.
Simulation seulement. Entrées mesurées sans vitesses : eccfit_<rel>.csv (α, β) et finfer_<rel>.csv (f_in).
À lancer depuis la racine du dépôt après « blind » : python validation_joint/promotion_study.py [dr3|dr4]"""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy.stats import norm
from wbt_pipeline import select, hr_band, MG_SUN
import jointlik as jl

t0 = time.time(); REL = sys.argv[1] if len(sys.argv) > 1 else "dr3"
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]; CC = CFG["calibration"]["contamination"]
GRID = "dr4" if REL == "dr4" else "dr3"
K = pd.read_pickle(f"K_{REL}.pkl"); hb, hs, _ = hr_band(K)
if P.get("hr_intercept") is not None: hb = P["hr_intercept"]
kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
m, _ = select(K, (hb, hs, P["hr_height"]), **kw); S = K[m].reset_index(drop=True)
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
dmod = 5*np.log10(S.d.values/10)
G = dict(x=S.x.values, vc=S.vc.values, sv=np.stack([S.cv_aa, S.cv_ad, S.cv_dd, S.fd], 1), d=S.d.values, rsky=S.rsky.values, M=S.M.values,
         m1=mass(S.MG1.values), m2=mass(S.MG2.values), G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
E = pd.read_csv(f"eccfit_{REL}.csv"); F = pd.read_csv(f"finfer_{REL}.csv")
row = lambda h, z: E[(E.gabarits == h) & (E.zone == z)].iloc[0]
A = {z: (row("newton", z).alpha_min, row("newton", z).alpha, row("newton", z).alpha_max) for z in ("bas", "haut")}
B = {z: (row("beta_separations", z).alpha_min, row("beta_separations", z).alpha, row("beta_separations", z).alpha_max) for z in ("bas", "haut")}
f = {r.shape: (r.f_bas, r.f_central, r.f_haut) for r in F.itertuples()}
a_c, b_c = (A["bas"][1], A["haut"][1]), (B["bas"][1], B["haut"][1])
CONT = [("raghavan2010", 0.0), ("raghavan2010", f["raghavan2010"][1]), ("raghavan2010", f["raghavan2010"][2]), ("mdwarf_pokemon", f["mdwarf_pokemon"][1])]
rng = np.random.default_rng(909)
zones1 = {"bas": np.where(G["x"] < 0.3)[0], "haut": np.where(G["x"] > 1)[0]}
T = jl.templates(G, zones1, b_c, CONT, CC, rng, R=30, vtmax=P["vtmax"], dvsky=P["dvsky"], grid=GRID)
print(f"gabarits : {T['D'].shape} ({time.time()-t0:.0f} s)", flush=True)

def dataset(Gs, zs, a, b, g, fin, shape, mb=1.0):
    lv, ps, k = jl.simulate_obs_cond(Gs, a, b, g, fin, shape, CC, rng, P["vtmax"], P["dvsky"], mass_bias=mb, grid=GRID)
    Sv = np.median(10**lv[zs["haut"]][k[zs["haut"]]])/np.median(10**lv[zs["bas"]][k[zs["bas"]]]) - 1
    return jl.lam(lv, ps, k, zs, T), Sv

out = []
for scale in (1, 2):
    idx = np.tile(np.arange(len(G["x"])), scale); Gs = jl.sub(G, np.arange(len(G["x"])), scale)
    zs = {"bas": np.where(Gs["x"] < 0.3)[0], "haut": np.where(Gs["x"] > 1)[0]}
    shrink = lambda band, c: c + (band - c)/np.sqrt(scale)                       # bandes mesurées resserrées en 1/√N
    fr = f["raghavan2010"][1]; n = 300
    LN, SN = np.array([dataset(Gs, zs, a_c, b_c, 0, fr, "raghavan2010") for _ in range(n)]).T
    LM, SM = np.array([dataset(Gs, zs, a_c, b_c, 1, fr, "raghavan2010") for _ in range(n)]).T
    thrN = LN.mean() + norm.isf(0.00135)*LN.std(); thrM = LM.mean() - norm.isf(0.00135)*LM.std()
    # test S, règle v0.8.1 : coins les plus défavorables (α, β, contamination haute pour Newton)
    aN = (shrink(A["bas"][2], A["bas"][1]), shrink(A["haut"][0], A["haut"][1])); bN = (shrink(B["bas"][2], B["bas"][1]), max(shrink(B["haut"][0], B["haut"][1]), shrink(B["bas"][2], B["bas"][1])))
    aM = (shrink(A["bas"][0], A["bas"][1]), shrink(A["haut"][2], A["haut"][1])); bM = (shrink(B["bas"][0], B["bas"][1]), shrink(B["haut"][2], B["haut"][1]))
    sN = np.array([dataset(Gs, zs, aN, bN, 0, f["raghavan2010"][2], "raghavan2010")[1] for _ in range(150)])
    sM = np.array([dataset(Gs, zs, aM, bM, 1, 0.0, "raghavan2010")[1] for _ in range(150)])
    S_thrN = sN.mean() + norm.isf(0.00135)*sN.std(); S_thrM = sM.mean() - norm.isf(0.00135)*sM.std()
    res = dict(echelle=scale, N=len(Gs["x"]),
               conj_MOND_rej_si_Newton=float(np.mean(LN < thrM)), conj_Newton_rej_si_MOND=float(np.mean(LM > thrN)),
               S_MOND_rej_si_Newton=float(np.mean(SN < S_thrM)), S_Newton_rej_si_MOND=float(np.mean(SM > S_thrN)),
               sep_conj_sigma=float((LM.mean() - LN.mean())/(0.5*(LN.std() + LM.std()))))
    if scale == 1:      # robustesse : biais de Λ sous Newton, en écarts-types de Λ
        for lab, kw2 in (("masse +20 %", dict(mb=1.2)), ("f_in 0,5 (hors bande)", dict(fin=0.5)),
                         ("β haut +0,3 (hors central)", dict(b=(b_c[0], b_c[1] + 0.3)))):
            args = dict(a=a_c, b=b_c, g=0, fin=fr, shape="raghavan2010", mb=1.0); args.update(kw2)
            Lx = np.array([dataset(Gs, zs, args["a"], args["b"], 0, args["fin"], args["shape"], args["mb"])[0] for _ in range(200)])
            res[f"biais_{lab}"] = float((Lx.mean() - LN.mean())/LN.std())
    out.append(res); print(json.dumps(res, ensure_ascii=False), f"({time.time()-t0:.0f} s)", flush=True)
R = pd.DataFrame(out); R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"promotion_{REL}.csv"), index=False)
r2 = out[-1]; r1 = out[0]
gain = max(r2["conj_MOND_rej_si_Newton"] - r2["S_MOND_rej_si_Newton"], r2["conj_Newton_rej_si_MOND"] - r2["S_Newton_rej_si_MOND"])
biais = max(abs(v) for k, v in r1.items() if k.startswith("biais_"))
print(f"CRITÈRE (a) gain ≥ 0,10 : {gain:+.2f} → {'OK' if gain >= 0.10 else 'NON'} ; (b) biais < 0,5σ : {biais:.2f} → {'OK' if biais < 0.5 else 'NON'}")
print("Critères (a) et (b) seulement, à titre de caractérisation. La promotion exigeait aussi (c) et (d) (stability_study.py) ;")
print("elle a été refusée le 18/10/2026 et la question est CLOSE (config decision.promotion.cloture) : S reste le test principal.")
