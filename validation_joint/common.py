"""v0.9.3 — Entrées communes des études de simulation du test conjoint (stabilité, origine de la puissance) et provenance.
Remplace le découpage de source par exec (audit externe du 19/10/2026, point D) ; méthode scientifique inchangée.
Portée exacte : aucune vitesse observée n'entre dans la statistique calculée sur les jeux simulés ; en revanche, la géométrie
utilisée (séparations, masses, bruits, distances) est celle de l'échantillon principal, dont la sélection applique des coupes
cinématiques (ṽ < vtmax, σ(ṽ), ΔRV, Δv sur le ciel) : elle dépend donc des vitesses observées (audit, point H)."""
import os, json, hashlib, subprocess, datetime
import numpy as np, pandas as pd
from wbt_pipeline import select, hr_band, MG_SUN

def load_inputs(REL):
    CFG = json.load(open("config_prereg.json", encoding="utf-8")); P = CFG["primary"]; CC = dict(CFG["calibration"]["contamination"])
    CC["screen"] = CC.get("screen_by_release", {}).get(REL, CC["screen"])
    K = pd.read_pickle(f"K_{REL}.pkl"); hb, hs, _ = hr_band(K)
    if P.get("hr_intercept") is not None: hb = P["hr_intercept"]
    kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
    m, _ = select(K, (hb, hs, P["hr_height"]), **kw); S = K[m].reset_index(drop=True)
    mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
    dmod = 5*np.log10(S.d.values/10)
    G = dict(x=S.x.values, vc=S.vc.values, sv=np.stack([S.cv_aa, S.cv_ad, S.cv_dd, S.fd], 1), d=S.d.values, rsky=S.rsky.values,
             M=S.M.values, m1=mass(S.MG1.values), m2=mass(S.MG2.values), G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
    E = pd.read_csv(f"eccfit_{REL}.csv"); F = pd.read_csv(f"finfer_{REL}.csv")
    row = lambda h, z: E[(E.gabarits == h) & (E.zone == z)].iloc[0]
    f = {r.shape: (r.f_bas, r.f_central, r.f_haut) for r in F.itertuples()}
    return dict(P=P, CC=CC, GRID="dr4" if REL == "dr4" else "dr3", R=CFG["calibration"]["joint"]["R_templates"], G=G, N=len(S),
                zs={"bas": np.where(G["x"] < 0.3)[0], "haut": np.where(G["x"] > 1)[0]},
                a_c=(row("newton", "bas").alpha, row("newton", "haut").alpha),
                b_c=(row("beta_separations", "bas").alpha, row("beta_separations", "haut").alpha), f=f,
                CONT=[("raghavan2010", 0.0), ("raghavan2010", f["raghavan2010"][1]), ("raghavan2010", f["raghavan2010"][2]),
                      ("mdwarf_pokemon", f["mdwarf_pokemon"][1])])

def _sha(p):
    if not os.path.exists(p): return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def provenance(REL, script, seeds, out_path, **extra):
    """Écrit à côté de la sortie un fichier JSON : date, commit, état du dépôt, graines, empreintes des entrées et du code."""
    import jointlik as jl
    def git(*a):
        try: return subprocess.run(["git", *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception: return None
    files = [f"K_{REL}.pkl", f"eccfit_{REL}.csv", f"finfer_{REL}.csv", "jointlik.py", "config_prereg.json", "validation_joint/common.py", script]
    rec = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), release=REL, script=script, commit=git("rev-parse", "HEAD"),
               depot_modifie=bool(git("status", "--porcelain", "--untracked-files=no")), graines=seeds,
               jointlik=dict(H_LV=jl.H_LV, H_PS=jl.H_PS, EPS=jl.EPS, n_eta=len(jl.ETA), pas_eta=float(jl.ETA[1] - jl.ETA[0]), n_alpha_fin=len(jl.AF)),
               numpy=np.__version__, pandas=pd.__version__, sha256={p: _sha(p) for p in files}, **extra)
    json.dump(rec, open(os.path.splitext(out_path)[0] + "_provenance.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return rec
