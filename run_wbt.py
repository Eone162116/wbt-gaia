"""Orchestrateur du WBT pré-enregistré.
  python run_wbt.py <dr3|dr4> fetch       téléchargement + vérification contre COUNT(*) serveur
  python run_wbt.py <dr3|dr4> build       paires, cinématique, crible DR4 (flags des candidats)
  python run_wbt.py <dr3|dr4> blind       échantillon principal, diagnostics AVEUGLES, calibration par simulations
  python run_wbt.py <dr3|dr4> freeze      manifeste SHA-256 (code, config, échantillon, simulations)
  python run_wbt.py <dr3|dr4> unblind     vérifie le gel, lève l'aveuglement, applique la règle de décision
  python run_wbt.py <dr3|dr4> multiverse  analyses secondaires (seulement après la levée)
  python run_wbt.py <dr3|dr4> all         enchaîne fetch, build, blind (s'arrête avant le gel)"""
import glob, itertools, json, os, sys, time, numpy as np, pandas as pd
from wbt_pipeline import (load_catalog, find_links, build_pairs, kinematics, hr_band, select, binned_test, BASE)
import calibrate as cal, blind as bl

CFG = json.load(open("config_prereg.json", encoding="utf-8"))
REL = sys.argv[1]; STAGE = sys.argv[2]
P = CFG["primary"]; ST = CFG["statistic"]
CODE = ["wbt_pipeline.py", "calibrate.py", "blind.py", "run_wbt.py", "fetch_gaia.py", "epoch_accel.py", "contamination.py",
        "finfer.py", "validation_contamination/pgrid.npy", "validation_contamination/pgrid_dr3.npy", "config_prereg.json"]
def frozen_files():
    mes = CFG["calibration"]["contamination"].get("f_in_mode") == "mesure"
    return CODE + [f"mocks_{REL}.csv", f"envelope_{REL}.csv"] + ([f"finfer_{REL}.csv"] if mes else [])
pattern = f"{REL}_*.csv" if glob.glob(f"{REL}_*.csv") else "s40_*.csv"      # s40_* = tranches DR3 de développement

def log(*a):
    msg = " ".join(str(x) for x in a); print(msg, flush=True)
    open(f"log_{REL}.txt", "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + msg + "\n")

def primary_selection(K):
    b, slope, _ = hr_band(K)
    if P["hr_intercept"] is not None: b = P["hr_intercept"]
    kw = {k: P[k] for k in ["dmax", "smin", "smax", "vtmax", "svt", "ruwe", "drv", "dvsky", "sph", "hr", "ipd", "degroup", "screen", "epoch"]}
    m, vt = select(K, (b, slope, P["hr_height"]), **kw)
    return m, vt, b

if STAGE in ("fetch", "all"):
    import fetch_gaia as fg
    fg.fetch_catalog(REL); log("catalogue complet :", fg.check_complete(REL))

if STAGE in ("build", "all"):
    df = load_catalog(pattern); I, J = find_links(df, snr_main=P["snr_main"], plx_main=1000/(1.05*max(CFG["multiverse"]["dmax"])))
    Pp = build_pairs(df, I, J, snr_main=P["snr_main"]); K = kinematics(df, Pp)
    log(f"{len(df)} étoiles, {len(I)} liens, {len(Pp)} paires mutuelles ; crible DR4 étoile : {df.scr4_star.mean():.3f}")
    if REL == "dr4":                      # flags de all_source_flags pour les seuls candidats plausibles
        import fetch_gaia as fg
        cand = (K.rsky < 30000) & (K.d < max(CFG["multiverse"]["dmax"])) & K.both_rv
        ids = np.concatenate([K.sid1[cand], K.sid2[cand]])
        F = fg.fetch_extra(REL, ids, CFG["flags_table"]["table"], CFG["flags_table"]["columns"]).set_index("source_id")
        ok = (F.non_single_star.fillna(0) == 0) & (F.sea_env_flag.fillna(0) == 0)
        K["flags_ok"] = ok.reindex(K.sid1).fillna(False).values & ok.reindex(K.sid2).fillna(False).values
        log(f"flags récupérés pour {len(F)} sources ; paires candidates propres : {K.flags_ok[cand].mean():.3f}")
    K.to_pickle(f"K_{REL}.pkl")

if STAGE == "epoch" or (STAGE == "all" and REL == "dr4"):
    if REL != "dr4": raise SystemExit("astrométrie par époque disponible seulement pour DR4")
    import epoch_accel as ea
    K = pd.read_pickle(f"K_{REL}.pkl"); E = CFG["epoch"]; G = CFG["multiverse"]
    # candidates = union des sélections les plus larges du multivers, hors crible par époques (≈ 12 000 composantes sur DR3)
    hb, hs, _ = hr_band(K); cand = np.zeros(len(K), bool)
    for sph in (False, True):
        mm, _ = select(K, (hb, hs, 0.8), hr=False, sph=sph, dmax=max(G["dmax"]), ruwe=max(G["ruwe"]),
                       degroup=None, screen="dr3", ipd=False, epoch=False)
        cand |= mm
    ids = np.unique(np.concatenate([K.sid1[cand], K.sid2[cand]]))
    log(f"crible par époques : téléchargement pour {len(ids)} composantes (≈ {len(ids)*0.15/1000:.1f} Go en RAW)")
    raw = ea.load_epoch_file(os.environ["WBT_EPOCH_FILE"]) if os.environ.get("WBT_EPOCH_FILE") else ea.fetch_epoch_dr4(ids)
    St = ea.test_all(ea.transits_from_table(raw)); St.to_csv(f"epoch_stats_{REL}.csv", index=False)
    bad = set(St.source_id[ea.flag(St, E["p_acc"], E["uwe_max"]) | (St.n_transits < E["min_transits"])])
    have = set(St.source_id); ok = lambda s: (s in have) and (s not in bad)
    K["epoch_ok"] = [ok(a) and ok(b) for a, b in zip(K.sid1, K.sid2)]
    log(f"crible par époques : {len(ids)} composantes candidates, {len(have)} avec données, {len(bad)} signalées ;"
        f" paires candidates retenues : {K.epoch_ok[cand].mean():.3f}")
    K.to_pickle(f"K_{REL}.pkl")

if STAGE in ("blind", "all"):
    K = pd.read_pickle(f"K_{REL}.pkl"); m, vt, b = primary_selection(K)
    if P.get("epoch") and "epoch_ok" not in K.columns: log("crible par époques absent (étape epoch non lancée ou DR3) : ignoré")
    S = K[m]; x = S.x.values; vc = S.vc.values; sv = (S.svt*S.vc).values; N = len(S)
    vt_blind = bl.blind(vt[m])
    log(f"[AVEUGLE] spécification principale : N = {N} (cible {CFG['decision']['n_target']}, minimum {CFG['decision']['n_min_powered']})"
        f" ; HR intercept {b:.3f} ; r/rM : {np.sum(x < ST['x_low'])} bas, {np.sum(x > ST['x_high'])} hauts")
    B, _ = binned_test(x, vt_blind)
    log("[AVEUGLE] médianes binées (permutées, sans information physique) :", B[["lo", "hi", "n", "med"]].round(3).to_dict("records"))
    log(f"[AVEUGLE] S permuté = {cal.step_stat(x, vt_blind, ST['x_low'], ST['x_high']):+.3f} (attendu ≈ 0)")
    pd.DataFrame(dict(sid1=S.sid1, sid2=S.sid2, x=x, M=S.M.values, vc=vc, sv=sv)).to_csv(f"sample_{REL}.csv", index=False)
    C = CFG["calibration"]; t = time.time()
    mocks = cal.calibrate(x, vc, sv, n_mocks=C["n_mocks"], seed=C["seed"], ecc=C["eccentricity"],
                          stat_kw=dict(x_low=ST["x_low"], x_high=ST["x_high"]), vtmax=P["vtmax"], dvsky=P["dvsky"])
    mocks.to_csv(f"mocks_{REL}.csv", index=False)
    sep = (mocks.mond.mean()-mocks.newton.mean())/mocks.newton.std()
    log(f"calibration : {C['n_mocks']} simulations ({time.time()-t:.0f} s) ; S Newton {mocks.newton.mean():+.3f} ± {mocks.newton.std():.3f}"
        f" ; S MOND {mocks.mond.mean():+.3f} ± {mocks.mond.std():.3f} ; séparation attendue {sep:.1f} σ")
    from wbt_pipeline import MG_SUN
    _mass = lambda MG: (lambda mm: np.where(mm < 0.7, mm + 0.05, mm))(10**(0.0725*(MG_SUN - MG)))
    dmod = 5*np.log10(S.d.values/10)
    geo = dict(x=x, vc=vc, sv=sv, d=S.d.values, rsky=S.rsky.values, M=S.M.values, m1=_mass(S.MG1.values), m2=_mass(S.MG2.values),
               G1=S.MG1.values + dmod, G2=S.MG2.values + dmod)
    CC = dict(C["contamination"]); t2 = time.time()
    CC["screen"] = CC.get("screen_by_release", {}).get(REL, CC["screen"])
    scenarios = None
    if CC.get("f_in_mode") == "mesure":            # v0.6 : f_in mesuré sur les drapeaux des paires candidates, sans vitesses
        import finfer as fi
        hb, hs, _ = hr_band(K)
        if P["hr_intercept"] is not None: hb = P["hr_intercept"]
        mc, _ = select(K, (hb, hs, P["hr_height"]), dmax=P["dmax"], smin=P["smin"], smax=P["smax"], svt=P["svt"], sph=P["sph"],
                       hr=P["hr"], degroup=P["degroup"], screen="dr3", epoch=False, ipd=False, ruwe=1e9, vtmax=1e9, dvsky=1e9, drv=1e9)
        Cd = K[mc]
        if REL == "dr4" and "epoch_ok" in Cd.columns: flags, grid = (~Cd.epoch_ok.astype(bool)).values, "dr4"
        else: flags, grid = (Cd.ruwe_max > 1.25).values, "dr3"
        band, r, sr = fi.infer_band(flags, _mass(Cd.MG1.values), _mass(Cd.MG2.values), Cd.d.values, Cd.rsky.values, CC["shapes"],
                                    grid, CC["r0"][grid], CC["nsig"], seed=C["seed"], stability=CC["stability"], q_range=CC["q_range"])
        scenarios = [(sh, f) for sh, fs in band.items() for f in fs]
        pd.DataFrame([dict(shape=sh, f_bas=fs[0], f_central=fs[1], f_haut=fs[2], taux=r, taux_err=sr, n=len(flags), drapeau=grid)
                      for sh, fs in band.items()]).to_csv(f"finfer_{REL}.csv", index=False)
        log(f"f_in mesuré (drapeau {grid}, sans vitesses) : taux signalé {r:.3f} ± {sr:.3f} sur {len(flags)} paires candidates ; "
            + " ; ".join(f"{sh} {fs[0]:.2f}–{fs[2]:.2f}" for sh, fs in band.items()))
    env = cal.calibrate_envelope(geo, n_per_f=CC["n_mocks_per_f"], seed=C["seed"], stat_kw=dict(x_low=ST["x_low"], x_high=ST["x_high"]),
                                 cfg=CC, vtmax=P["vtmax"], dvsky=P["dvsky"], scenarios=scenarios)
    env.to_csv(f"envelope_{REL}.csv", index=False)
    g = env.groupby(["shape", "f_in"]).S
    log(f"enveloppe de contamination ({CC['screen']}, {time.time()-t2:.0f} s) : S newtonien moyen : "
        + ", ".join(f"{sh[:8]} {f:.2f} → {v:+.3f}" for (sh, f), v in g.mean().items())
        + f" ; séparation MOND vs pire Newton : {(mocks.mond.mean() - g.mean().max())/mocks.newton.std():.1f} σ")

def frozen_arrays():
    s = pd.read_csv(f"sample_{REL}.csv")
    return dict(sample_ids=np.sort(np.concatenate([s.sid1.values, s.sid2.values]).astype(np.int64)))

if STAGE == "freeze":
    files = frozen_files()
    man = bl.freeze(f"FREEZE_{REL}.json", files, frozen_arrays())
    log("gel enregistré :", man["frozen_at_utc"], "manifeste", man["manifest_sha256"][:16])

if STAGE == "unblind":
    man = bl.verify(f"FREEZE_{REL}.json", frozen_files(), frozen_arrays())
    K = pd.read_pickle(f"K_{REL}.pkl"); m, vt, b = primary_selection(K)
    x = K.x.values[m]; v = vt[m]
    S_obs = cal.step_stat(x, v, ST["x_low"], ST["x_high"])
    res = cal.decide(S_obs, pd.read_csv(f"mocks_{REL}.csv"), CFG["decision"]["p_threshold"], env=pd.read_csv(f"envelope_{REL}.csv"))
    B, s = binned_test(x, v)
    res.update(N=int(m.sum()), binned_log10_newton_over_mond=float(np.log10(s["pref_newton"])), unblinded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               freeze_manifest=man["manifest_sha256"])
    if res["N"] < CFG["decision"]["n_min_powered"]: res["verdict"] += " — étude déclarée sous-dimensionnée"
    json.dump(res, open(f"RESULT_{REL}.json", "w", encoding="utf-8"), indent=2, default=float)
    log("[LEVÉE]", json.dumps(res, indent=1, default=float, ensure_ascii=False))

if STAGE == "multiverse":
    if not (os.path.exists(f"RESULT_{REL}.json") and os.path.exists(f"FREEZE_{REL}.json")):
        raise SystemExit("multivers interdit avant la levée de l'aveuglement")
    man = bl.verify(f"FREEZE_{REL}.json", frozen_files(), frozen_arrays())
    if json.load(open(f"RESULT_{REL}.json", encoding="utf-8")).get("freeze_manifest") != man["manifest_sha256"]:
        raise SystemExit("multivers interdit : RESULT ne correspond pas au gel en cours (fichier étranger ou périmé)")
    K = pd.read_pickle(f"K_{REL}.pkl"); b, slope, _ = hr_band(K)
    if P["hr_intercept"] is not None: b = P["hr_intercept"]
    G = CFG["multiverse"]; rows = []
    for combo in itertools.product(*G.values()):
        spec = dict(zip(G.keys(), combo)); hrh = spec.pop("hr")
        if REL == "dr3" and (spec["screen"] == "dr4" or spec.get("epoch")): continue
        m, vt = select(K, (b, slope, hrh or 0.8), hr=hrh is not None, ipd=P["ipd"], **spec)
        x, v = K.x.values[m], vt[m]
        rows.append(dict(**spec, hr=hrh, n=int(m.sum()), S=cal.step_stat(x, v, ST["x_low"], ST["x_high"])))
    R = pd.DataFrame(rows); R.to_csv(f"multiverse_{REL}.csv", index=False)
    log(f"multivers : {len(R)} spécifications ; S médian {R.S.median():+.3f} ; part S > +0,075 : {(R.S > 0.075).mean():.2f}")
