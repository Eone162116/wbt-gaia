"""v0.7-B — Choix du seuil σ(ṽ) par simulation seulement (aucune vitesse observée n'est utilisée).
Pour chaque seuil, l'échantillon est défini par les seules coupes indépendantes des vitesses de la spécification principale
(pas de coupe ṽ, Δv_sky ni ΔRV : les simulations appliquent elles-mêmes ṽ < 2,5 et Δv_sky < 10 km/s).
Mesures : effectifs, marche S attendue sous Newton (pur et contaminé, excentricité thermique) et sous MOND
(thermique et superthermique α 1,0→1,6, le pire cas pour MOND), séparation dans le pire cas.
À lancer depuis la racine du dépôt après « build » : python validation_calibration/sigma_threshold_study.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from wbt_pipeline import select, hr_band, MG_SUN
import calibrate as cal, contamination as ct

REL = sys.argv[1] if len(sys.argv) > 1 else "dr3"
CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]; CC = dict(CFG["calibration"]["contamination"])
CC["screen"] = CC.get("screen_by_release", {}).get(REL, CC.get("screen")); CC["shape"] = "raghavan2010"
K = pd.read_pickle(f"K_{REL}.pkl"); hb, hs, _ = hr_band(K)
if P.get("hr_intercept") is not None: hb = P["hr_intercept"]
mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
R = int(os.environ.get("WBT_STUDY_R", "1000")); F_CONT = float(os.environ.get("WBT_STUDY_F", "0.376"))
rng = np.random.default_rng(7); rows = []
for thr in (0.1, 0.15, 0.2, 0.3):
    kw = {k: P[k] for k in ["dmax", "smin", "smax", "ruwe", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
    m, _ = select(K, (hb, hs, P["hr_height"]), svt=thr, vtmax=1e9, dvsky=1e9, drv=1e9, **kw)
    S = K[m]; x, vc, rsky = S.x.values, S.vc.values, S.rsky.values
    sv = np.stack([S.cv_aa, S.cv_ad, S.cv_dd, S.fd], 1)
    dmod = 5*np.log10(S.d.values/10)
    geo = dict(x=x, vc=vc, sv=sv, d=S.d.values, rsky=rsky, M=S.M.values, m1=mass(S.MG1.values), m2=mass(S.MG2.values),
               G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
    res = {}
    for lab, truth, ecc in (("N", "newton", "thermal"), ("M", "mond", "thermal"), ("M16", "mond", (1.0, 1.6))):
        v = np.concatenate([cal.simulate_batch(x, vc, sv, truth, rng, 50, P["vtmax"], P["dvsky"], ecc, rsky) for _ in range(R//50)])
        res[lab] = (np.nanmean(v), np.nanstd(v))
    vc_ = np.array([cal.step_stat(*ct.contaminated_newton(geo, F_CONT, rng, CC, P["vtmax"], P["dvsky"], cal.kepler_geometry)) for _ in range(R)])
    res["Nc"] = (np.nanmean(vc_), np.nanstd(vc_))
    sd = 0.5*(res["N"][1] + res["M"][1])
    worst = (res["M16"][0] - max(res["N"][0], res["Nc"][0]))/(0.5*(max(res["Nc"][1], res["N"][1]) + res["M16"][1]))
    rows.append(dict(seuil=thr, N=int(m.sum()), n_bas=int((x < 0.3).sum()), n_haut=int((x > 1).sum()),
                     S_newton=res["N"][0], S_newton_cont=res["Nc"][0], S_mond=res["M"][0], S_mond_a16=res["M16"][0],
                     sd_newton=res["N"][1], sd_mond=res["M"][1], separation=(res["M"][0] - res["N"][0])/sd, separation_pire=worst))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sigma_threshold_{REL}.csv"), index=False)
pd.set_option("display.width", 220); print(T.round(3).to_string(index=False))
